"""
ListOps -- Nested list-operation tree evaluation, V4-style circuit
==================================================================

Fifth compositional generalization benchmark in the paper. Direct parallel
to CruxMini: same architectural approach (tied integer distribution +
learned bilinear op lookup table), different operation set, different
held-out op for the compositional split.

## Grammar

Binary operation trees of depth up to 3:
    expr  := lit | binop
    lit   := integer in [0, 9]
    binop := expr OP expr    (OP in {MIN, MAX, SUM_MOD, PROD_MOD})

All operations act over integers in [0, 9]:
  - MIN(a, b)     = min(a, b)                      (no modulo needed)
  - MAX(a, b)     = max(a, b)                      (no modulo needed)
  - SUM_MOD(a, b) = (a + b) mod 10
  - PROD_MOD(a, b)= (a * b) mod 10

## The compositional generalization test

Training split: trees where MAX appears ONLY as a leaf-level op, i.e.
  MAX has LITERAL children only.
MIN, SUM_MOD, PROD_MOD can appear at any depth in training.

Test split (test_max): trees where MAX has at least one NON-literal child.
The model must generalize max to compound subexpression operands despite
never seeing `MAX(MIN(3,5), 7)` or `MAX(SUM(2,4), 8)` in training.

This is the ListOps analog of:
  - SCAN: hold out JUMP compositions
  - CruxMini: hold out `*` compositions
  - ListOps: hold out MAX compositions

Same architectural test, different task class. If V4's content-structure
separation is a general property, it should transfer from arithmetic
(CruxMini `*`) to comparison + aggregation (ListOps MAX).

## Architecture (V4-style, same skeleton as CruxMini)

Slot state per node = distribution over integer values (B, N, n_ints).
Leaves are initialized to one-hot of their literal value. Op nodes update
via a learned (n_ops, n_ints, n_ints, n_ints) lookup table.

`op_table[op, i, j, k] = logit that (i OP j) = k`

Small random init, everything learned end-to-end. Content (integer
identity) flows as soft distributions; structure (op_type) indexes the
lookup slice. Never any MLP on content.

Param count: n_ops × n_ints³ = 4 × 10³ = 4,000 learned params.
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


# ---------------------------------------------------------------------------
# Grammar constants
# ---------------------------------------------------------------------------

N_INTS = 10          # integers [0, 9]
OP_MIN = 0
OP_MAX = 1
OP_SUM = 2
OP_PROD = 3
N_OPS = 4
OP_NAMES = ['min', 'max', 'sum', 'prod']

NO_CHILD = -1
MAX_NODES = 31       # depth-3 full binary tree has 15 nodes; 31 = depth-4 safe buffer

# Node types: 0 = literal, 1 = binop
NODE_LIT = 0
NODE_OP = 1

# Which op is held out for compositional generalization
HELD_OUT_OP = OP_MAX


def apply_op(op, a, b):
    if op == OP_MIN:
        return min(a, b)
    if op == OP_MAX:
        return max(a, b)
    if op == OP_SUM:
        return (a + b) % 10
    if op == OP_PROD:
        return (a * b) % 10
    raise ValueError(op)


# ---------------------------------------------------------------------------
# Tree generation
# ---------------------------------------------------------------------------

def _gen_subtree(depth_budget, rng, nodes, force_leaf_held_out):
    """Recursively grow a subtree rooted at a new node. Returns (node_id, value).

    If force_leaf_held_out is True, any HELD_OUT_OP (MAX) that gets chosen must
    have literal children only. Other ops always recurse normally. This lets us
    generate the training split where MAX appears only at leaf level.
    """
    my_id = len(nodes)
    # Choose node type
    if depth_budget <= 0 or rng.random() < 0.35:
        v = rng.randint(0, N_INTS - 1)
        nodes.append({'type': NODE_LIT, 'value': v, 'op': 0,
                      'left': NO_CHILD, 'right': NO_CHILD})
        return my_id, v

    nodes.append(None)  # placeholder for recursion
    op = rng.choice([OP_MIN, OP_MAX, OP_SUM, OP_PROD])   # always include all 4

    if op == HELD_OUT_OP and force_leaf_held_out:
        # force literal children for the held-out op
        left_id, left_v = _gen_subtree(0, rng, nodes, force_leaf_held_out)
        right_id, right_v = _gen_subtree(0, rng, nodes, force_leaf_held_out)
    else:
        left_id, left_v = _gen_subtree(depth_budget - 1, rng, nodes, force_leaf_held_out)
        right_id, right_v = _gen_subtree(depth_budget - 1, rng, nodes, force_leaf_held_out)

    nodes[my_id] = {
        'type': NODE_OP,
        'value': 0,
        'op': op,
        'left': left_id,
        'right': right_id,
    }
    v = apply_op(op, left_v, right_v)
    nodes[my_id]['value'] = v
    return my_id, v


def has_nested_held_out(nodes):
    """True if the tree contains HELD_OUT_OP (MAX) where at least one child is an op."""
    for n in nodes:
        if n is None:
            continue
        if n['type'] == NODE_OP and n['op'] == HELD_OUT_OP:
            lc = nodes[n['left']]
            rc = nodes[n['right']]
            if lc['type'] == NODE_OP or rc['type'] == NODE_OP:
                return True
    return False


def has_any_held_out(nodes):
    return any(
        n['type'] == NODE_OP and n['op'] == HELD_OUT_OP
        for n in nodes if n is not None
    )


def generate_tree(depth, split, rng):
    """Generate a random tree meeting the split constraint.

    split='train':
        Trees where MAX appears only with literal children (leaf-level max).
        Other ops can appear at any depth. Trees may still contain max.
    split='test_max':
        Trees where MAX has at least one NON-literal child. The model must
        generalize max to compound subexpressions even though it only saw
        leaf-level max in training.
    split='all':
        Any valid tree.
    """
    attempts = 0
    while attempts < 500:
        attempts += 1
        nodes = []
        # Force leaf-level held-out op in training; allow everything else in test.
        force_leaf = (split == 'train')
        _gen_subtree(depth, rng, nodes, force_leaf)
        if split == 'train':
            # Safety: reject anything that accidentally has nested held-out op
            if has_nested_held_out(nodes):
                continue
            return nodes
        elif split == 'test_max':
            if has_nested_held_out(nodes):
                return nodes
            else:
                continue
        else:
            return nodes
    raise RuntimeError(f'failed to generate tree for split={split}')


def make_listops_batch(batch_size, split, depth=3, device='cpu'):
    """Build a batch of trees. Returns a dict of tensors."""
    rng = random.Random(None)  # per-batch fresh
    cats = []       # (B, N): 0 = lit, 1 = op
    ops = []        # (B, N): 0-3 for ops, ignored for lits
    lits = []       # (B, N): 0-9 for lits, ignored for ops
    left = []
    right = []
    mask = []
    targets = []    # (B,) final answer at root

    for _ in range(batch_size):
        nodes = generate_tree(depth, split, rng)
        n = len(nodes)
        c = [0] * MAX_NODES
        o = [0] * MAX_NODES
        lv = [0] * MAX_NODES
        l = [0] * MAX_NODES
        r = [0] * MAX_NODES
        m = [False] * MAX_NODES
        for i, node in enumerate(nodes):
            if i >= MAX_NODES:
                break
            c[i] = node['type']
            o[i] = node['op']
            lv[i] = node['value'] if node['type'] == NODE_LIT else 0
            l[i] = max(node['left'], 0)
            r[i] = max(node['right'], 0)
            m[i] = True
        cats.append(c)
        ops.append(o)
        lits.append(lv)
        left.append(l)
        right.append(r)
        mask.append(m)
        targets.append(nodes[0]['value'])  # root is node 0

    return {
        'cats': torch.tensor(cats, dtype=torch.long, device=device),
        'ops': torch.tensor(ops, dtype=torch.long, device=device),
        'lits': torch.tensor(lits, dtype=torch.long, device=device),
        'left': torch.tensor(left, dtype=torch.long, device=device),
        'right': torch.tensor(right, dtype=torch.long, device=device),
        'mask': torch.tensor(mask, dtype=torch.bool, device=device),
        'targets': torch.tensor(targets, dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------------------
# ListOps V4-style Circuit
# ---------------------------------------------------------------------------

@dataclass
class ListOpsConfig:
    d_inner: int = 32
    max_nodes: int = MAX_NODES
    n_ints: int = N_INTS
    n_ops: int = N_OPS
    n_passes: int = 4             # enough for depth 3 + slack

    lr: float = 5e-3
    warmup_steps: int = 300
    n_steps: int = 5000
    batch_size: int = 128
    grad_clip: float = 1.0
    eval_every: int = 250
    n_eval: int = 500


class ListOpsCircuit(nn.Module):
    """V4-style circuit for list-ops tree evaluation.

    Slot state per node = distribution over integer values (B, N, n_ints).
    Leaves are initialized to one-hot of their literal value. Op nodes
    update via a learned (n_ops, n_ints, n_ints, n_ints) lookup table.

    Identical structure to CruxMini circuit, different op set.
    The op_table is the content-structure separator: structure (op_type)
    indexes the lookup slice; content (integer distributions) combines
    via bilinear outer product against the slice.
    """
    def __init__(self, cfg: ListOpsConfig):
        super().__init__()
        self.cfg = cfg
        # Learned op lookup: (n_ops, n_ints, n_ints, n_ints)
        # op_table[op, i, j, k] = logit that "(i OP j) = k"
        self.op_table = nn.Parameter(torch.randn(
            cfg.n_ops, cfg.n_ints, cfg.n_ints, cfg.n_ints
        ) * 0.1)

    def _init_state(self, cats, lits, mask):
        """Literals -> one-hot over ints. Non-literals -> zeros (overwritten later)."""
        B, N = cats.shape
        device = cats.device
        state = torch.zeros(B, N, self.cfg.n_ints, device=device)
        lit_mask = (cats == NODE_LIT) & mask
        lit_onehot = F.one_hot(lits.clamp(0, self.cfg.n_ints - 1),
                               num_classes=self.cfg.n_ints).float()
        state = torch.where(lit_mask.unsqueeze(-1), lit_onehot, state)
        return state

    def _op_step(self, state, cats, ops, left, right, mask, update_mask,
                 return_logits=False):
        B, N, n_ints = state.shape
        n_ops = self.cfg.n_ops

        # Gather children's distributions
        l_exp = left.unsqueeze(-1).expand(-1, -1, n_ints)
        r_exp = right.unsqueeze(-1).expand(-1, -1, n_ints)
        left_dist = torch.gather(state, 1, l_exp)
        right_dist = torch.gather(state, 1, r_exp)

        # Soft lookup: pick the op_table slice per node
        slice_table = self.op_table[ops.clamp(0, n_ops - 1)]

        # logits[k] = sum_{i,j} left[i] * right[j] * table[i, j, k]
        node_logits = torch.einsum('bni,bnj,bnijk->bnk', left_dist, right_dist, slice_table)
        op_dist = F.softmax(node_logits, dim=-1)

        new_state = torch.where(update_mask.unsqueeze(-1), op_dist, state)
        if return_logits:
            return new_state, node_logits
        return new_state

    def forward(self, batch):
        cfg = self.cfg
        B, N = batch['cats'].shape

        cats = batch['cats']
        ops = batch['ops']
        lits = batch['lits']
        left = batch['left'].clamp(0, N - 1)
        right = batch['right'].clamp(0, N - 1)
        mask = batch['mask']

        state = self._init_state(cats, lits, mask)
        op_mask = (cats == NODE_OP) & mask

        # Bottom-up passes
        for _ in range(cfg.n_passes - 1):
            state = self._op_step(state, cats, ops, left, right, mask, op_mask)
        # Final pass returns raw logits for proper cross_entropy at the root
        state, node_logits = self._op_step(
            state, cats, ops, left, right, mask, op_mask, return_logits=True
        )

        # Root readout from node 0
        op_node_logits = node_logits[:, 0, :]
        root_is_lit = (cats[:, 0] == NODE_LIT).unsqueeze(-1).float()
        lit_logits = state[:, 0, :] * 10.0
        root_logits = root_is_lit * lit_logits + (1 - root_is_lit) * op_node_logits
        return root_logits


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def get_lr(step, warmup, total, base_lr, min_lr=1e-5):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cos = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr + (base_lr - min_lr) * cos


def accuracy(logits_or_dist, targets):
    preds = logits_or_dist.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def evaluate(model, n_eval, device):
    model.eval()
    results = {}
    with torch.no_grad():
        b = make_listops_batch(n_eval, split='train', device=device)
        results['acc_train'] = accuracy(model(b), b['targets'])
        b = make_listops_batch(n_eval, split='test_max', device=device)
        results['acc_test_max'] = accuracy(model(b), b['targets'])
    model.train()
    return results


def train_listops(model, cfg, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    model.train()

    history = []
    for step in range(1, cfg.n_steps + 1):
        lr = get_lr(step, cfg.warmup_steps, cfg.n_steps, cfg.lr)
        for g in opt.param_groups:
            g['lr'] = lr

        batch = make_listops_batch(cfg.batch_size, split='train', device=device)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch['targets'])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                tr_acc = accuracy(logits, batch['targets'])
            print(f"  [listops] step {step:4d}/{cfg.n_steps}  loss={loss.item():.4f}  "
                  f"train_batch={tr_acc:.3f}  lr={lr:.2e}", flush=True)

        if step % cfg.eval_every == 0:
            ev = evaluate(model, cfg.n_eval, device)
            print(f"  [listops] eval @ step {step}  train={ev['acc_train']:.3f}  "
                  f"test_max={ev['acc_test_max']:.3f}", flush=True)
            history.append({'step': step, **ev})

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--n-steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = ListOpsConfig(
        lr=args.lr,
        n_steps=500 if args.smoke else args.n_steps,
        eval_every=100 if args.smoke else 250,
        n_eval=200 if args.smoke else 500,
    )

    print("=" * 66)
    print("ListOps -- Nested list-op tree evaluation (V4-style circuit)")
    print("=" * 66)
    print(f"  Grammar: binary op trees, ops in {{min, max, sum_mod, prod_mod}}, ints in [0,9]")
    print(f"  Train: trees where MAX has LITERAL children only")
    print(f"  Test:  trees where MAX has at least one NON-literal child")
    print(f"  Arch:  tied integer distribution + learned op lookup table")
    print(f"  seed={args.seed}  n_steps={cfg.n_steps}  batch={cfg.batch_size}  lr={cfg.lr}")
    print("=" * 66, flush=True)

    model = ListOpsCircuit(cfg).to(device)
    p = count_params(model)
    print(f"\nListOps circuit: {p:,} params\n", flush=True)

    t0 = time.time()
    history = train_listops(model, cfg, device)
    t_train = time.time() - t0

    final = evaluate(model, cfg.n_eval * 2, device)

    print("\n" + "=" * 66)
    print("FINAL RESULTS")
    print("=" * 66)
    print(f"  Train acc (leaf-level MAX):         {final['acc_train']:.3f}")
    print(f"  Test_max acc (compound MAX):        {final['acc_test_max']:.3f}")
    print(f"  Random baseline (10 classes):       0.100")
    print(f"  Training time:                      {t_train:.1f}s")
    print("=" * 66)

    out_tag = f'listops_s{args.seed}'
    out_dir = Path(f'results/{out_tag}')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump({
            'final': final,
            'history': history,
            'params': p,
            'config': {
                'd_inner': cfg.d_inner,
                'n_passes': cfg.n_passes,
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
