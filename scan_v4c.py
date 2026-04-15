"""
SCAN V4c -- Learned positional ops for compositional generalization
====================================================================

V4b proved the architectural insight (positional slot state + tied action
embedding) with a near-symbolic 164-param implementation. V4c is the fair
paper story: keep the positional state and tied embedding, but REPLACE the
hand-crafted modifier/combinator gather/scatter with LEARNED ops.

The learning signal is in a small "position-attention generator" that maps
(child_count, subtype) -> a positional attention matrix. The attention is
then applied LINEARLY to the child buffer(s) -- so content (action
embeddings) flows through as linear combinations of the child vectors, never
through a content-dependent nonlinearity.

This is the content/structure separation spelled out explicitly:
  - Attention weights = STRUCTURE (depend on count + subtype, not content)
  - Buffer values     = CONTENT   (action embeddings, flow through linearly)

For the modifier:
    attn[p, q] = softmax(generator([count_child, subtype_onehot])[p, q])
    output[p] = sum_q attn[p, q] * child_buffer[q]
    count_new = child_count * repeat(subtype)    (hardcoded count factor)

For the combinator:
    concat_buf = cat([left_buf, right_buf], dim=positions)   # (B, N, 2*MO, d)
    attn[p, q] = softmax(generator([lc, rc, subtype_onehot])[p, q])
    output[p] = sum_q attn[p, q] * concat_buf[q]
    count_new = left_count + right_count

Count transforms stay hardcoded because (a) they're trivial features and
(b) we want to isolate the "learned position remap" claim from "learned
count arithmetic" which is a separate hard problem.

Usage:
    python scan_v4c.py              # full 5000 steps
    python scan_v4c.py --smoke      # quick 500 steps
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
class V4CConfig:
    d_inner: int = 32
    d_hidden: int = 128     # hidden dim of attention generator
    max_nodes: int = MAX_NODES
    max_output_len: int = MAX_OUTPUT_LEN
    n_actions: int = N_ACTIONS

    lr: float = 3e-3
    warmup_steps: int = 200
    n_steps: int = 5000
    batch_size: int = 64
    grad_clip: float = 1.0
    eval_every: int = 250
    n_eval: int = 500


# ---------------------------------------------------------------------------
# Learned position-attention ops
# ---------------------------------------------------------------------------

class LearnedModifierOp(nn.Module):
    """Modifier op with LEARNED position attention.

    The attention generator takes (count_norm, subtype_onehot) as input and
    outputs a flat (max_out * max_out) logits tensor. These are softmax'd
    over input positions and applied linearly to the child buffer.
    """
    def __init__(self, max_out, d_hidden, n_sub=2):
        super().__init__()
        self.max_out = max_out
        self.n_sub = n_sub

        self.gen = nn.Sequential(
            nn.Linear(1 + n_sub, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, max_out * max_out),
        )
        # Residual init toward IDENTITY attention (output[p] = child[p])
        # by biasing the last layer toward an identity pattern.
        with torch.no_grad():
            self.gen[-1].weight.zero_()
            eye = torch.eye(max_out).flatten() * 4.0  # strong identity pull
            self.gen[-1].bias.copy_(eye)

        # Count factors per subtype: twice -> 2, thrice -> 3
        self.register_buffer('count_factors', torch.tensor([2.0, 3.0]))

    def forward(self, child_buf, child_count, subs):
        """child_buf:   (B, N, MO, d)
           child_count: (B, N)   float
           subs:        (B, N)   long in {0, 1}
        """
        B, N, MO, d = child_buf.shape
        device = child_buf.device

        # Conditioning features
        cc_norm = child_count.unsqueeze(-1) / float(MO)           # (B, N, 1)
        sub_oh = F.one_hot(subs.clamp(0, self.n_sub - 1),
                           num_classes=self.n_sub).float()         # (B, N, n_sub)
        feats = torch.cat([cc_norm, sub_oh], dim=-1)              # (B, N, 1+n_sub)

        logits = self.gen(feats).view(B, N, MO, MO)               # (B, N, MO, MO)
        attn = F.softmax(logits, dim=-1)                           # over input positions

        # Linear combination: output[p] = sum_q attn[p, q] * child[q]
        new_buf = torch.einsum('bnpq,bnqd->bnpd', attn, child_buf)

        # Count update (hardcoded factor)
        factors = self.count_factors[subs.clamp(0, self.n_sub - 1)]
        new_count = (child_count * factors).clamp(max=float(MO))
        return new_buf, new_count


class LearnedCombinatorOp(nn.Module):
    """Combinator op with LEARNED position attention over concatenated children.

    The attention generator takes (left_count_norm, right_count_norm,
    subtype_onehot) -> flat (max_out * 2*max_out) logits. Softmax is over
    the 2*max_out input positions (first half = left child, second half =
    right child). Applied linearly to the concatenated buffer.
    """
    def __init__(self, max_out, d_hidden, n_sub=2):
        super().__init__()
        self.max_out = max_out
        self.n_sub = n_sub

        self.gen = nn.Sequential(
            nn.Linear(2 + n_sub, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, max_out * 2 * max_out),
        )
        # Residual init toward [output[p] = left[p]] (AND-like identity).
        with torch.no_grad():
            self.gen[-1].weight.zero_()
            bias = torch.zeros(max_out, 2 * max_out)
            for p in range(max_out):
                bias[p, p] = 4.0  # pull attention to left[p] by default
            self.gen[-1].bias.copy_(bias.flatten())

    def forward(self, left_buf, left_count, right_buf, right_count, subs):
        """
        left_buf:    (B, N, MO, d)
        left_count:  (B, N)
        right_buf:   (B, N, MO, d)
        right_count: (B, N)
        subs:        (B, N) long in {0, 1}  (0=AND, 1=AFTER)
        """
        B, N, MO, d = left_buf.shape
        device = left_buf.device

        lc_norm = left_count.unsqueeze(-1) / float(MO)
        rc_norm = right_count.unsqueeze(-1) / float(MO)
        sub_oh = F.one_hot(subs.clamp(0, self.n_sub - 1),
                           num_classes=self.n_sub).float()
        feats = torch.cat([lc_norm, rc_norm, sub_oh], dim=-1)      # (B, N, 2 + n_sub)

        logits = self.gen(feats).view(B, N, MO, 2 * MO)             # (B, N, MO, 2*MO)
        attn = F.softmax(logits, dim=-1)

        concat_buf = torch.cat([left_buf, right_buf], dim=2)        # (B, N, 2*MO, d)
        new_buf = torch.einsum('bnpq,bnqd->bnpd', attn, concat_buf) # (B, N, MO, d)

        new_count = (left_count + right_count).clamp(max=float(MO))
        return new_buf, new_count


# ---------------------------------------------------------------------------
# Full V4c circuit
# ---------------------------------------------------------------------------

class SCANCircuitV4C(nn.Module):
    """Positional-slot circuit with LEARNED modifier/combinator ops.

    Architectural commitments (identical to V4b except the ops are learned):
      1. Tied action_embed for primitive init + decoder output projection
      2. Positional slot buffers (MO, d_inner)
      3. Content flows as linear combinations (attention applied to buffers)
      4. Structure (attention patterns) is learned from (count, subtype) input

    Only the learned ops have significant parameters. action_embed is 5*d_inner,
    position embedding is max_out*d_inner, LearnedModifierOp is the attention
    generator (~few K params), LearnedCombinatorOp is similar.
    """
    def __init__(self, cfg: V4CConfig):
        super().__init__()
        self.cfg = cfg
        MO = cfg.max_output_len
        d = cfg.d_inner

        # Tied action embedding
        self.action_embed = nn.Embedding(cfg.n_actions, d)

        # Learned ops
        self.mod_op = LearnedModifierOp(MO, cfg.d_hidden, n_sub=2)
        self.comb_op = LearnedCombinatorOp(MO, cfg.d_hidden, n_sub=2)

    def _init_primitive_state(self, subs, mask):
        """Primitive nodes: buffer[0] = action_embed[subtype + 1], zeros elsewhere."""
        B, N = subs.shape
        d = self.cfg.d_inner
        MO = self.cfg.max_output_len

        action_ids = (subs + 1).clamp(0, self.cfg.n_actions - 1)
        prim_vecs = self.action_embed(action_ids)                   # (B, N, d)

        buf = torch.zeros(B, N, MO, d, device=subs.device)
        buf[:, :, 0, :] = prim_vecs * mask.unsqueeze(-1).float()
        count = mask.float()                                         # 1 if primitive, else 0
        return buf, count

    def forward(self, batch):
        cfg = self.cfg
        device = batch['node_cats'].device
        B = batch['node_cats'].shape[0]
        N = cfg.max_nodes
        MO = cfg.max_output_len
        d = cfg.d_inner

        cats = batch['node_cats']
        subs = batch['node_subs']
        mask = batch['node_mask']
        cl = batch['child_left'].clamp(0, N - 1)
        cr = batch['child_right'].clamp(0, N - 1)

        prim_mask = (cats == CAT_PRIMITIVE) & mask
        mod_mask = (cats == CAT_MODIFIER) & mask
        comb_mask = (cats == CAT_COMBINATOR) & mask

        # PASS 1: primitives
        buffers, counts = self._init_primitive_state(subs, prim_mask)

        # PASS 2: modifiers pull from their single child
        cl_exp = cl.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, MO, d)
        child_buf = torch.gather(buffers, 1, cl_exp)                   # (B, N, MO, d)
        child_cnt = torch.gather(counts, 1, cl)                        # (B, N)
        mod_buffers, mod_counts = self.mod_op(child_buf, child_cnt, subs)

        buffers = torch.where(mod_mask.unsqueeze(-1).unsqueeze(-1),
                              mod_buffers, buffers)
        counts = torch.where(mod_mask, mod_counts, counts)

        # PASS 3: combinators pull from both children
        cl_exp = cl.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, MO, d)
        cr_exp = cr.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, MO, d)
        left_buf = torch.gather(buffers, 1, cl_exp)
        right_buf = torch.gather(buffers, 1, cr_exp)
        left_cnt = torch.gather(counts, 1, cl)
        right_cnt = torch.gather(counts, 1, cr)

        comb_buffers, comb_counts = self.comb_op(
            left_buf, left_cnt, right_buf, right_cnt, subs
        )
        buffers = torch.where(comb_mask.unsqueeze(-1).unsqueeze(-1),
                              comb_buffers, buffers)
        counts = torch.where(comb_mask, comb_counts, counts)

        # Root readout
        root_buf = buffers[:, 0, :, :]                                 # (B, MO, d)
        logits = root_buf @ self.action_embed.weight.T                 # (B, MO, n_actions)

        # STOP bias beyond active length
        root_count = counts[:, 0:1]                                    # (B, 1)
        positions = torch.arange(MO, device=device).unsqueeze(0)       # (1, MO)
        is_stop_pos = (positions >= root_count).unsqueeze(-1).float()  # (B, MO, 1)
        stop_bias = torch.zeros(cfg.n_actions, device=device)
        stop_bias[STOP] = 8.0
        logits = logits + is_stop_pos * stop_bias
        return logits


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def get_lr(step, warmup, total, base_lr, min_lr=1e-5):
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


def train_v4c(model, cfg, device):
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
            print(f"  [v4c] step {step:4d}/{cfg.n_steps}  loss={loss.item():.4f}  "
                  f"train={train_acc:.3f}  lr={lr:.2e}", flush=True)

        if step % cfg.eval_every == 0:
            ev = evaluate(model, cfg.n_eval, device)
            print(f"  [v4c] eval @ step {step}  train={ev['acc_train']:.3f}  "
                  f"jump_comp={ev['acc_jump_comp']:.3f}", flush=True)
            history.append({'step': step, **ev})

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--d-inner', type=int, default=32)
    parser.add_argument('--d-hidden', type=int, default=128)
    parser.add_argument('--n-steps', type=int, default=5000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = V4CConfig(
        d_inner=args.d_inner,
        d_hidden=args.d_hidden,
        lr=args.lr,
        n_steps=500 if args.smoke else args.n_steps,
        batch_size=args.batch,
        eval_every=100 if args.smoke else 250,
        n_eval=200 if args.smoke else 500,
    )

    print("=" * 66)
    print("SCAN V4c -- Learned positional ops for compositional generalization")
    print("=" * 66)
    print(f"  d_inner={cfg.d_inner}  d_hidden={cfg.d_hidden}")
    print(f"  n_steps={cfg.n_steps}  batch={cfg.batch_size}  lr={cfg.lr}")
    print("  Architecture:")
    print("    * Positional slot state (MO, d_inner)")
    print("    * Tied action_embed (primitive init + decoder output proj)")
    print("    * LEARNED modifier op: attn = gen(count_child, subtype)")
    print("    * LEARNED combinator op: attn = gen(l_count, r_count, subtype)")
    print("    * Content flows as LINEAR combinations (no content MLPs)")
    print("=" * 66, flush=True)

    model = SCANCircuitV4C(cfg).to(device)
    p = count_params(model)
    print(f"\nSCAN V4c circuit: {p:,} params\n", flush=True)

    t0 = time.time()
    history = train_v4c(model, cfg, device)
    t_train = time.time() - t0

    final = evaluate(model, cfg.n_eval * 2, device)

    print("\n" + "=" * 66)
    print("FINAL RESULTS")
    print("=" * 66)
    print(f"  Train acc:                {final['acc_train']:.3f}")
    print(f"  Jump compositions (OOD):  {final['acc_jump_comp']:.3f}")
    print(f"  Training time:            {t_train:.1f}s")
    print("=" * 66)
    print("\nReference points:")
    print(f"  V4b (symbolic, 164 params):   train 1.000 / jump 1.000")
    print(f"  Tree TF small (846K params):  train 1.000 / jump 0.204 avg")
    print(f"  Tree TF big (6.5M params):    train 1.000 / jump 0.064 avg")
    print()

    out_tag = f'scan_v4c_d{cfg.d_inner}_h{cfg.d_hidden}_s{args.seed}'
    out_dir = Path(f'results/{out_tag}')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump({
            'final': final,
            'history': history,
            'params': p,
            'config': {
                'd_inner': cfg.d_inner,
                'd_hidden': cfg.d_hidden,
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
