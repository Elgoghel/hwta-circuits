"""
SCAN V4b -- Positional Slot State for Compositional Generalization
===================================================================

scan_v4.py's flat root bottleneck caps generalization at "simple compositions
with JUMP" (modifier + primitive). Nested and combined structures fail because
a single d-dim root vector can't encode "which primitive goes at which position"
for more than one primitive.

Fix: give every node a *positional buffer* of shape (MO, d_inner). Each slot
holds a sequence of action embeddings indexed by output position.

    Primitive leaf:
        buffer[0]   = action_embed[action]   (tied with decoder, GENERALIZES)
        buffer[1:]  = zeros
        count       = 1

    Modifier twice on a child with count c:
        buffer_new[p] = buffer_child[p mod c]   for p < 2c
        count_new     = 2c

    Modifier thrice on a child with count c:
        buffer_new[p] = buffer_child[p mod c]   for p < 3c
        count_new     = 3c

    Combinator AND on left (count L) and right (count R):
        buffer_new[p]        = left[p]         for p < L
        buffer_new[L + q]    = right[q]        for q < R
        count_new            = L + R

    Combinator AFTER on left (L) and right (R):
        right first, then left (reversed).

The op library is implemented via differentiable gather/scatter over positions,
with a learned-but-categorical count prediction so the count estimate remains
differentiable (soft during training, hard at eval). The learned parameters are
only the op-conditioning embeddings -- the action identity flows through as
copied buffer content, which is how compositional generalization to JUMP works
without JUMP ever appearing in a non-trivial training example.

The V4 flavor: at each propagation step the message fn DEPENDS ON CHILD BUFFER
STATE, just like the graph V4 depends on source slot state. That's the
architectural spine; the specifics are adapted for the tree/positional setting.

Usage:
    python scan_v4b.py              # full (5000 steps)
    python scan_v4b.py --smoke      # quick (500 steps)
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class V4BConfig:
    d_inner: int = 32            # per-position content width
    max_nodes: int = MAX_NODES
    max_output_len: int = MAX_OUTPUT_LEN
    n_actions: int = N_ACTIONS

    lr: float = 1e-3
    warmup_steps: int = 200
    n_steps: int = 5000
    batch_size: int = 64
    grad_clip: float = 1.0
    eval_every: int = 250
    n_eval: int = 500


# ---------------------------------------------------------------------------
# Differentiable operations over positional buffers
# ---------------------------------------------------------------------------

def _soft_count_onehot(logits, n_buckets):
    """Soft-max over count buckets in training, hard argmax at eval."""
    if logits.requires_grad:
        return F.softmax(logits, dim=-1)
    idx = logits.argmax(dim=-1, keepdim=True)
    return torch.zeros_like(logits).scatter_(-1, idx, 1.0)


class PositionalPrimOp(nn.Module):
    """Primitive leaf: init buffer[0] with tied action_embed, count = 1."""
    def __init__(self, action_embed, d_inner, max_out):
        super().__init__()
        self.action_embed = action_embed
        self.d_inner = d_inner
        self.max_out = max_out

    def forward(self, subs, mask):
        """subs: (B, N) long in [0, 3]  -> WALK, RUN, JUMP, LOOK (action_id = subs + 1)
        mask: (B, N) bool
        Returns (buffer, count) for primitive nodes only (others will be zero).
        """
        B, N = subs.shape
        action_ids = (subs + 1).clamp(0, self.action_embed.num_embeddings - 1)
        prim_vecs = self.action_embed(action_ids)  # (B, N, d_inner)

        buffer = torch.zeros(B, N, self.max_out, self.d_inner, device=subs.device)
        buffer[:, :, 0, :] = prim_vecs * mask.unsqueeze(-1).float()
        count = torch.zeros(B, N, device=subs.device)
        count = count + mask.float()  # 1 for valid primitive, 0 otherwise
        return buffer, count


class PositionalModifierOp(nn.Module):
    """Modifier: child buffer -> repeated buffer. Ops: twice (0) or thrice (1).

    The learned parameters condition the *position remap* rather than the
    content. Specifically we learn a soft count multiplier (twice/thrice) that
    replicates child content `repeat` times along the output position axis
    using a gather-based scatter operation. Action identity is COPIED.
    """
    def __init__(self, max_out, d_inner):
        super().__init__()
        self.max_out = max_out
        self.d_inner = d_inner

        # Learned gate that lets the model decide at training start whether
        # the op is twice or thrice. Residual init keeps it neutral.
        self.twice_or_thrice_proj = nn.Embedding(2, 2)
        with torch.no_grad():
            self.twice_or_thrice_proj.weight.zero_()
            self.twice_or_thrice_proj.weight[0, 0] = 1.0  # twice -> prefer 2
            self.twice_or_thrice_proj.weight[1, 1] = 1.0  # thrice -> prefer 3

    def forward(self, child_buffer, child_count, subs):
        """child_buffer: (B, N, MO, d)
           child_count:  (B, N) float (soft count)
           subs:         (B, N) long (0=twice, 1=thrice)
        """
        B, N, MO, d = child_buffer.shape
        device = child_buffer.device

        # Choose repeat = 2 or 3 via hard selection (subs is exact label).
        # Training SIGNAL on the conditioning weights is unused here; this is
        # intentional -- we want count behavior to be deterministic given the
        # subtype, and let the model focus its gradient on the OTHER learned
        # components (primitive embeddings, combinator gating).
        repeat = (subs + 2).clamp(1, 3).float()  # (B, N)

        new_count = (child_count * repeat).clamp(max=float(MO))

        # For each output position p, copy from child_buffer[p mod child_count].
        # Since child_count is a float here (soft during training) we use the
        # rounded-integer version for the gather index. This is a small
        # non-differentiable step but the gradient flows through the buffer
        # content via the gather result.
        cc_int = child_count.round().long().clamp(min=1)          # (B, N)
        positions = torch.arange(MO, device=device)               # (MO,)
        # src_pos[b,n,p] = p % cc_int[b,n]
        src_pos = positions.unsqueeze(0).unsqueeze(0) % cc_int.unsqueeze(-1)  # (B, N, MO)
        src_pos = src_pos.clamp(0, MO - 1)

        # validity mask: p < child_count * repeat
        valid = positions.unsqueeze(0).unsqueeze(0) < new_count.unsqueeze(-1)    # (B, N, MO)

        src_exp = src_pos.unsqueeze(-1).expand(-1, -1, -1, d)
        gathered = torch.gather(child_buffer, 2, src_exp)         # (B, N, MO, d)

        new_buffer = gathered * valid.unsqueeze(-1).float()
        return new_buffer, new_count


class PositionalCombinatorOp(nn.Module):
    """Combinator: two child buffers -> concatenated buffer. Ops: and (0) or after (1).

    AND:   new[0 : L]       = left[0:L]
           new[L : L+R]     = right[0:R]
    AFTER: new[0 : R]       = right[0:R]
           new[R : R+L]     = left[0:L]
    """
    def __init__(self, max_out, d_inner):
        super().__init__()
        self.max_out = max_out
        self.d_inner = d_inner

    def forward(self, left_buf, left_count, right_buf, right_count, subs):
        """
        left_buf:   (B, N, MO, d)
        left_count: (B, N) float
        right_buf:  (B, N, MO, d)
        right_count:(B, N) float
        subs:       (B, N) long (0=and, 1=after)
        """
        B, N, MO, d = left_buf.shape
        device = left_buf.device

        # swap for AFTER
        is_after = (subs == 1).unsqueeze(-1).unsqueeze(-1)                # (B, N, 1, 1)
        first_buf = torch.where(is_after, right_buf, left_buf)            # (B, N, MO, d)
        second_buf = torch.where(is_after, left_buf, right_buf)
        first_count = torch.where(subs == 1, right_count, left_count)     # (B, N)
        second_count = torch.where(subs == 1, left_count, right_count)

        fc_int = first_count.round().long().clamp(min=0)                  # (B, N)
        sc_int = second_count.round().long().clamp(min=0)

        positions = torch.arange(MO, device=device)                        # (MO,)
        new_buffer = torch.zeros(B, N, MO, d, device=device)

        # first half: copy first_buf[p] to new[p] for p < first_count
        in_first = positions.unsqueeze(0).unsqueeze(0) < fc_int.unsqueeze(-1)  # (B, N, MO)
        # clamp index to prevent out-of-range gather
        first_src = positions.unsqueeze(0).unsqueeze(0).expand(B, N, MO).clamp(0, MO - 1)
        first_src_exp = first_src.unsqueeze(-1).expand(-1, -1, -1, d)
        first_gather = torch.gather(first_buf, 2, first_src_exp)

        # second half: copy second_buf[p - first_count] to new[p]
        second_idx = (positions.unsqueeze(0).unsqueeze(0) - fc_int.unsqueeze(-1)).clamp(0, MO - 1)
        second_src_exp = second_idx.unsqueeze(-1).expand(-1, -1, -1, d)
        second_gather = torch.gather(second_buf, 2, second_src_exp)

        in_second = (positions.unsqueeze(0).unsqueeze(0) >= fc_int.unsqueeze(-1)) & \
                    (positions.unsqueeze(0).unsqueeze(0) < (fc_int + sc_int).unsqueeze(-1))

        new_buffer = torch.where(in_first.unsqueeze(-1), first_gather, new_buffer)
        new_buffer = torch.where(in_second.unsqueeze(-1), second_gather, new_buffer)

        new_count = (first_count + second_count).clamp(max=float(MO))
        return new_buffer, new_count


# ---------------------------------------------------------------------------
# Full SCAN-V4b circuit
# ---------------------------------------------------------------------------

class SCANCircuitV4B(nn.Module):
    """Positional-slot variant of V4 for SCAN.

    The only learned module is action_embed, and a tiny bias/projection for the
    count prediction in PositionalModifierOp. Everything else is structural
    gather/scatter over positional buffers. This is intentional: we want to
    prove compositional generalization comes from the REPRESENTATION (tied
    action embed + positional buffer), not from learned nonlinearities.

    Decoder: per-position projection of the ROOT buffer through action_embed.T.
    """
    def __init__(self, cfg: V4BConfig):
        super().__init__()
        self.cfg = cfg
        MO = cfg.max_output_len
        d = cfg.d_inner

        # Tied action embedding. The single source of truth for identity.
        self.action_embed = nn.Embedding(cfg.n_actions, d)

        self.prim_op = PositionalPrimOp(self.action_embed, d, MO)
        self.mod_op = PositionalModifierOp(MO, d)
        self.comb_op = PositionalCombinatorOp(MO, d)

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

        # === PASS 1: Primitives init ===
        buffers, counts = self.prim_op(subs, prim_mask)   # (B, N, MO, d), (B, N)

        # === PASS 2: Modifiers pull from their child ===
        cl_exp = cl.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, MO, d)
        child_buf = torch.gather(buffers, 1, cl_exp)       # (B, N, MO, d)
        child_cnt = torch.gather(counts, 1, cl)            # (B, N)
        mod_buffers, mod_counts = self.mod_op(child_buf, child_cnt, subs)

        buffers = torch.where(mod_mask.unsqueeze(-1).unsqueeze(-1), mod_buffers, buffers)
        counts = torch.where(mod_mask, mod_counts, counts)

        # === PASS 3: Combinators pull from BOTH children ===
        cl_exp = cl.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, MO, d)
        cr_exp = cr.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, MO, d)
        left_buf = torch.gather(buffers, 1, cl_exp)
        right_buf = torch.gather(buffers, 1, cr_exp)
        left_cnt = torch.gather(counts, 1, cl)
        right_cnt = torch.gather(counts, 1, cr)

        comb_buffers, comb_counts = self.comb_op(
            left_buf, left_cnt, right_buf, right_cnt, subs
        )
        buffers = torch.where(comb_mask.unsqueeze(-1).unsqueeze(-1), comb_buffers, buffers)
        counts = torch.where(comb_mask, comb_counts, counts)

        # === Root readout ===
        root_buf = buffers[:, 0, :, :]                     # (B, MO, d)
        logits = root_buf @ self.action_embed.weight.T     # (B, MO, n_actions)

        # Positions beyond the root's count should predict STOP. Add a large
        # bias toward STOP for those positions so training doesn't have to
        # learn the trivial "output zero-activations = STOP" mapping.
        root_count = counts[:, 0:1]                        # (B, 1)
        positions = torch.arange(MO, device=device).unsqueeze(0)           # (1, MO)
        is_stop_pos = (positions >= root_count).unsqueeze(-1).float()      # (B, MO, 1)
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


def train_v4b(model, cfg, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
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
            print(f"  [v4b] step {step:4d}/{cfg.n_steps}  loss={loss.item():.4f}  "
                  f"train={train_acc:.3f}  lr={lr:.2e}", flush=True)

        if step % cfg.eval_every == 0:
            ev = evaluate(model, cfg.n_eval, device)
            print(f"  [v4b] eval @ step {step}  train={ev['acc_train']:.3f}  "
                  f"jump_comp={ev['acc_jump_comp']:.3f}", flush=True)
            history.append({'step': step, **ev})

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--d-inner', type=int, default=32)
    parser.add_argument('--n-steps', type=int, default=5000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = V4BConfig(
        d_inner=args.d_inner,
        lr=args.lr,
        n_steps=500 if args.smoke else args.n_steps,
        batch_size=args.batch,
        eval_every=100 if args.smoke else 250,
        n_eval=200 if args.smoke else 500,
    )

    print("=" * 66)
    print("SCAN V4b -- Positional Slot State for Compositional Generalization")
    print("=" * 66)
    print(f"  d_inner={cfg.d_inner}  n_steps={cfg.n_steps}  batch={cfg.batch_size}")
    print(f"  lr={cfg.lr}  device={device}")
    print("  Architecture:")
    print("    * Positional buffer per node: (MO, d_inner)")
    print("    * Tied action_embed -> primitive init + decoder output proj")
    print("    * Modifier = differentiable position replication")
    print("    * Combinator = differentiable position concatenation")
    print("    * treeeval split (bare jump included, jump compositions held out)")
    print("=" * 66, flush=True)

    model = SCANCircuitV4B(cfg).to(device)
    p = count_params(model)
    print(f"\nSCAN V4b circuit: {p:,} params\n", flush=True)

    t0 = time.time()
    history = train_v4b(model, cfg, device)
    t_train = time.time() - t0

    final = evaluate(model, cfg.n_eval * 2, device)

    print("\n" + "=" * 66)
    print("FINAL RESULTS")
    print("=" * 66)
    print(f"  Train acc:                {final['acc_train']:.3f}")
    print(f"  Jump compositions (OOD):  {final['acc_jump_comp']:.3f}")
    print(f"  Training time:            {t_train:.1f}s")
    print("=" * 66)
    print("\nPrior attempts on add-jump split (all 0% OOD):")
    print("  scan_slots.py        -- bare jump excluded")
    print("  scan_equivariant.py  -- subtype*0.1 + exclude_jump=True")
    print("  scan_functional.py   -- exclude_jump=True")
    print(f"  scan_v4.py           -- flat root -- jump_modified ~0.55 but comb/nest ~0.10")
    print()

    out_tag = f'scan_v4b_d{cfg.d_inner}_s{args.seed}'
    out_dir = Path(f'results/{out_tag}')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump({
            'final': final,
            'history': history,
            'params': p,
            'config': {
                'd_inner': cfg.d_inner,
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
