"""
Tree-input Transformer baseline for SCAN compositional generalization.

Takes the same pre-parsed tree input that scan_v4b uses (node categories +
subtypes + child pointers) and runs a standard transformer encoder over the
nodes, then decodes an action sequence from the root position. This is the
fair architectural comparison: if the transformer also hits 100% on jump
compositions, V4b's win is trivial; if it hits the well-known SCAN add-jump
ceiling (<20%), V4b's architectural choice matters.

Usage:
    python scan_tree_tf.py              # full
    python scan_tree_tf.py --smoke      # quick
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

from scan_slots import (
    STOP, WALK, RUN, JUMP, LOOK, N_ACTIONS,
    MAX_OUTPUT_LEN, compute_accuracy,
)
from scan_equivariant import (
    CAT_PRIMITIVE, CAT_MODIFIER, CAT_COMBINATOR, N_CATEGORIES,
)
from scan_treeeval import MAX_NODES, make_treeeval_batch


@dataclass
class TreeTFConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.0

    max_nodes: int = MAX_NODES
    max_output_len: int = MAX_OUTPUT_LEN
    n_actions: int = N_ACTIONS
    n_categories: int = N_CATEGORIES
    max_subtypes: int = 4

    lr: float = 3e-4
    warmup_steps: int = 200
    n_steps: int = 5000
    batch_size: int = 64
    grad_clip: float = 1.0
    eval_every: int = 250
    n_eval: int = 500


class TreeTransformer(nn.Module):
    """Vanilla transformer encoder over parsed SCAN trees.

    Node embedding = cat_embed + sub_embed + position_embed + child_edge_bias.
    After n_layers of self-attention, the root position is decoded via
    positional queries + cross-attention (same spirit as scan_v4.py's decoder).
    """
    def __init__(self, cfg: TreeTFConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.cat_embed = nn.Embedding(cfg.n_categories, d)
        self.sub_embed = nn.Embedding(cfg.max_subtypes, d)
        self.node_pos_embed = nn.Embedding(cfg.max_nodes, d)

        # Child structure signal: for each node, add an embedding that encodes
        # "I am parent of these nodes". We inject child identity via a learned
        # relative-position bias on the attention.
        self.child_embed = nn.Embedding(cfg.max_nodes + 1, d)  # +1 for "no child"

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

        # Output decoder: positional queries cross-attend over node states
        self.out_pos_embed = nn.Embedding(cfg.max_output_len, d)
        self.out_q = nn.Linear(d, d, bias=False)
        self.out_k = nn.Linear(d, d, bias=False)
        self.out_v = nn.Linear(d, d, bias=False)
        self.out_ln = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, cfg.n_actions)

    def forward(self, batch):
        cfg = self.cfg
        B = batch['node_cats'].shape[0]
        N = cfg.max_nodes
        d = cfg.d_model
        device = batch['node_cats'].device

        cats = batch['node_cats']  # (B, N)
        subs = batch['node_subs']
        mask = batch['node_mask']  # (B, N) bool
        cl = batch['child_left'].clamp(0, N - 1)
        cr = batch['child_right'].clamp(0, N - 1)

        # Input features
        node_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
        h = (self.cat_embed(cats)
             + self.sub_embed(subs.clamp(0, cfg.max_subtypes - 1))
             + self.node_pos_embed(node_ids)
             + self.child_embed(cl)
             + self.child_embed(cr))

        # Transformer encoder with src_key_padding_mask
        attn_mask_pad = ~mask  # True for padded positions
        h = self.encoder(h, src_key_padding_mask=attn_mask_pad)  # (B, N, d)

        # Output via positional query cross-attention
        positions = torch.arange(cfg.max_output_len, device=device)
        q = self.out_pos_embed(positions).unsqueeze(0).expand(B, -1, -1)  # (B, MO, d)
        q = self.out_q(q)
        k = self.out_k(h)
        v = self.out_v(h)

        scores = q @ k.transpose(-2, -1) / math.sqrt(d)
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        attn = F.softmax(scores, dim=-1)
        context = attn @ v  # (B, MO, d)
        context = self.out_ln(context)

        logits = self.out_proj(context)
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
        b_train = make_treeeval_batch(n_eval, split='train', device=device)
        logits = model(b_train)
        results['acc_train'] = compute_accuracy(
            logits, b_train['outputs'], b_train['output_mask']
        )

        b_jump = make_treeeval_batch(n_eval, split='test_jump', device=device)
        logits = model(b_jump)
        results['acc_jump_comp'] = compute_accuracy(
            logits, b_jump['outputs'], b_jump['output_mask']
        )
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

        batch = make_treeeval_batch(cfg.batch_size, split='train', device=device)
        logits = model(batch)

        loss_pp = F.cross_entropy(
            logits.reshape(-1, cfg.n_actions),
            batch['outputs'].reshape(-1),
            reduction='none',
        ).reshape(logits.shape[0], logits.shape[1])
        loss = (loss_pp * batch['output_mask']).sum() / batch['output_mask'].sum().clamp(min=1)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                train_acc = compute_accuracy(
                    logits, batch['outputs'], batch['output_mask']
                )
            print(f"  [tree-tf] step {step:4d}/{cfg.n_steps}  loss={loss.item():.4f}  "
                  f"train={train_acc:.3f}  lr={lr:.2e}", flush=True)

        if step % cfg.eval_every == 0:
            ev = evaluate(model, cfg.n_eval, device)
            print(f"  [tree-tf] eval @ step {step}  train={ev['acc_train']:.3f}  "
                  f"jump_comp={ev['acc_jump_comp']:.3f}", flush=True)
            history.append({'step': step, **ev})

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-steps', type=int, default=5000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = TreeTFConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        lr=args.lr,
        n_steps=500 if args.smoke else args.n_steps,
        batch_size=args.batch,
        eval_every=100 if args.smoke else 250,
        n_eval=200 if args.smoke else 500,
    )

    print("=" * 66)
    print("TREE TRANSFORMER -- SCAN add-jump baseline on parsed tree input")
    print("=" * 66)
    print(f"  d_model={cfg.d_model}  layers={cfg.n_layers}  heads={cfg.n_heads}")
    print(f"  n_steps={cfg.n_steps}  batch={cfg.batch_size}  lr={cfg.lr}")
    print("=" * 66, flush=True)

    model = TreeTransformer(cfg).to(device)
    p = count_params(model)
    print(f"\nTree Transformer: {p:,} params\n", flush=True)

    t0 = time.time()
    history = train(model, cfg, device)
    t_train = time.time() - t0

    final = evaluate(model, cfg.n_eval * 2, device)

    print("\n" + "=" * 66)
    print("FINAL RESULTS -- Tree Transformer")
    print("=" * 66)
    print(f"  Train acc:                {final['acc_train']:.3f}")
    print(f"  Jump compositions (OOD):  {final['acc_jump_comp']:.3f}")
    print(f"  Training time:            {t_train:.1f}s")
    print("=" * 66)

    out_tag = f'scan_tree_tf_d{cfg.d_model}_l{cfg.n_layers}_s{args.seed}'
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
                'n_heads': cfg.n_heads,
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
