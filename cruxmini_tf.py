"""
Tree-input Transformer baseline for CruxMini.

Same tree-structured input as cruxmini.py. If the transformer also hits 100%
on compound multiplication, the V4 circuit's win is trivial; if it collapses
the way it did on SCAN, the content-structure separation story extends to
arithmetic code execution.

Usage:
    python cruxmini_tf.py              # full
    python cruxmini_tf.py --smoke      # quick
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cruxmini import (
    N_INTS, N_OPS, MAX_NODES, NODE_LIT, NODE_OP,
    make_cruxmini_batch, accuracy,
)


@dataclass
class CruxMiniTFConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.0

    max_nodes: int = MAX_NODES
    n_ints: int = N_INTS
    n_ops: int = N_OPS

    lr: float = 5e-4
    warmup_steps: int = 300
    n_steps: int = 8000
    batch_size: int = 128
    grad_clip: float = 1.0
    eval_every: int = 250
    n_eval: int = 500


class TreeTransformer(nn.Module):
    """Vanilla transformer over cruxmini trees.

    Node features:
      - node type (lit vs op)
      - literal value (for lit nodes) or op id (for op nodes)
      - position in the tree
      - child pointers as learned relative embeddings
    """
    def __init__(self, cfg: CruxMiniTFConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.type_embed = nn.Embedding(2, d)                # lit=0, op=1
        self.lit_embed = nn.Embedding(cfg.n_ints, d)
        self.op_embed = nn.Embedding(cfg.n_ops, d)
        self.pos_embed = nn.Embedding(cfg.max_nodes, d)
        self.child_embed = nn.Embedding(cfg.max_nodes + 1, d)  # +1 sentinel

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=cfg.n_heads,
                dim_feedforward=d * cfg.ff_mult,
                dropout=cfg.dropout,
                batch_first=True,
                activation='gelu',
            ),
            num_layers=cfg.n_layers,
        )

        self.readout = nn.Linear(d, cfg.n_ints)

    def forward(self, batch):
        cfg = self.cfg
        B, N = batch['cats'].shape
        device = batch['cats'].device

        cats = batch['cats']
        ops = batch['ops']
        lits = batch['lits']
        left = batch['left']
        right = batch['right']
        mask = batch['mask']

        node_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

        h = (self.type_embed(cats)
             + self.pos_embed(node_ids)
             + self.lit_embed(lits.clamp(0, cfg.n_ints - 1))
             + self.op_embed(ops.clamp(0, cfg.n_ops - 1))
             + self.child_embed(left.clamp(0, cfg.max_nodes))
             + self.child_embed(right.clamp(0, cfg.max_nodes)))

        h = self.encoder(h, src_key_padding_mask=~mask)

        root = h[:, 0, :]
        logits = self.readout(root)
        return logits


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def get_lr(step, warmup, total, base_lr, min_lr=1e-6):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cos = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr + (base_lr - min_lr) * cos


def evaluate(model, n_eval, device):
    model.eval()
    results = {}
    with torch.no_grad():
        b = make_cruxmini_batch(n_eval, split='train', device=device)
        results['acc_train'] = accuracy(model(b), b['targets'])
        b = make_cruxmini_batch(n_eval, split='test_mul', device=device)
        results['acc_test_mul'] = accuracy(model(b), b['targets'])
    model.train()
    return results


def train(model, cfg, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    model.train()

    history = []
    for step in range(1, cfg.n_steps + 1):
        lr = get_lr(step, cfg.warmup_steps, cfg.n_steps, cfg.lr)
        for g in opt.param_groups:
            g['lr'] = lr

        batch = make_cruxmini_batch(cfg.batch_size, split='train', device=device)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch['targets'])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                tr_acc = accuracy(logits, batch['targets'])
            print(f"  [tree-tf] step {step:4d}/{cfg.n_steps}  loss={loss.item():.4f}  "
                  f"train_batch={tr_acc:.3f}  lr={lr:.2e}", flush=True)

        if step % cfg.eval_every == 0:
            ev = evaluate(model, cfg.n_eval, device)
            print(f"  [tree-tf] eval @ step {step}  train={ev['acc_train']:.3f}  "
                  f"test_mul={ev['acc_test_mul']:.3f}", flush=True)
            history.append({'step': step, **ev})

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-steps', type=int, default=8000)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = CruxMiniTFConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        lr=args.lr,
        n_steps=500 if args.smoke else args.n_steps,
        batch_size=args.batch,
        eval_every=100 if args.smoke else 250,
        n_eval=200 if args.smoke else 500,
    )

    print("=" * 66)
    print("CruxMini -- Tree Transformer baseline")
    print("=" * 66)
    print(f"  d_model={cfg.d_model}  layers={cfg.n_layers}  n_steps={cfg.n_steps}")
    print("=" * 66, flush=True)

    model = TreeTransformer(cfg).to(device)
    p = count_params(model)
    print(f"\nTree Transformer: {p:,} params\n", flush=True)

    t0 = time.time()
    history = train(model, cfg, device)
    t_train = time.time() - t0

    final = evaluate(model, cfg.n_eval * 2, device)

    print("\n" + "=" * 66)
    print("FINAL RESULTS -- Tree Transformer on CruxMini")
    print("=" * 66)
    print(f"  Train acc (leaf-level *):     {final['acc_train']:.3f}")
    print(f"  Test_mul acc (compound *):    {final['acc_test_mul']:.3f}")
    print(f"  Training time:                {t_train:.1f}s")
    print("=" * 66)
    print("\nReference: CruxMini V4 circuit (3000 params): 1.000 / 1.000")
    print()

    out_tag = f'cruxmini_tf_d{cfg.d_model}_l{cfg.n_layers}_s{args.seed}'
    out_dir = Path(f'results/{out_tag}')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump({
            'final': final,
            'history': history,
            'params': p,
            'config': {
                'd_model': cfg.d_model,
                'n_layers': cfg.n_layers,
                'n_steps': cfg.n_steps,
                'batch_size': cfg.batch_size,
                'lr': cfg.lr,
                'seed': args.seed,
            },
            'train_time': t_train,
        }, f, indent=2)
    print(f"Results saved to {out_dir / 'results.json'}")


if __name__ == '__main__':
    main()
