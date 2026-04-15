"""
CruxMini -- Nested arithmetic tree evaluation, V4-style circuit
================================================================

Stepping stone toward real CRUXEval. CRUXEval requires executing arbitrary
Python (strings, lists, loops, dicts, method calls) -- a 1-2 month effort.
CruxMini is a tractable code-reasoning testbed that captures the essential
challenge (compositional tree evaluation with content-dependent ops) in a
minimal grammar.

## Grammar

Arithmetic expression trees of depth up to 3:
    expr := lit | binop
    lit  := integer in [0, 9]
    binop := expr OP expr    (OP in {+, -, *})
    all arithmetic is mod 10 to keep outputs in [0, 9]

The tree is randomly generated with a target depth budget. At each node we
either bottom out into a literal or branch into a binary op.

## The compositional generalization test

Training split: trees where `*` appears ONLY as a leaf-level op, i.e.
`lit * lit`. Multiplication's left and right operands are always integer
literals in the training set.

Test split: trees where `*` has at least one non-leaf child. The model must
generalize multiplication to subexpression operands even though it never
saw `(3+2) * 4` or `(5-1) * (2+7)` in training. This is the direct code
analog of SCAN add-jump: "does the model learn the OP abstractly, or only
in the narrow contexts it saw?"

## The architecture (V4-style)

Each tree node owns a slot. Slot state is a d_inner-dim embedding of the
current integer value (or probability distribution over integers, in the
soft case).

Key commitments:

1. **Tied integer embedding.** `int_embed = nn.Embedding(10, d_inner)` is
   used for BOTH leaf initialization AND decoder projection. Just like
   SCAN's tied action_embed.

2. **Lookup-table ops.** Each binary op has a learned (10, 10, 10) logit
   tensor `op_table[op, left_int, right_int] -> result_int_dist`. At train
   time this acts as a soft lookup; at inference time we use argmax-ish.
   Content (integer identity) flows through as soft distributions, never
   through a content-dependent MLP. This is content-structure separation
   for arithmetic: the "structure" is `op_type`, the "content" is the
   integer operands, and they're combined via a lookup indexed by content.

3. **Bottom-up 3-pass propagation.** Literals are set at pass 0, then
   non-literal nodes update based on their children. Max depth is 3, so 3
   passes suffice.

Compare to a matched-input tree transformer on the same trees.
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
OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
N_OPS = 3
OP_NAMES = ['+', '-', '*']

NO_CHILD = -1
MAX_NODES = 31       # depth-3 full binary tree has 15 nodes; 31 = depth-4 safe buffer

# Node types: 0 = literal, 1 = binop
NODE_LIT = 0
NODE_OP = 1


def apply_op(op, a, b):
    if op == OP_ADD:
        return (a + b) % 10
    if op == OP_SUB:
        return (a - b) % 10
    if op == OP_MUL:
        return (a * b) % 10
    raise ValueError(op)


# ---------------------------------------------------------------------------
# Tree generation
# ---------------------------------------------------------------------------

def _gen_subtree(depth_budget, rng, nodes, force_leaf_mul):
    """Recursively grow a subtree rooted at a new node. Returns the node id.

    If force_leaf_mul is True, any `*` op that gets chosen must have literal
    children only. Add/sub always recurse normally. This lets us generate the
    training split where `*` appears only at leaf level, but add/sub can
    appear at any depth.
    """
    my_id = len(nodes)
    # Choose node type
    if depth_budget <= 0 or rng.random() < 0.35:
        v = rng.randint(0, N_INTS - 1)
        nodes.append({'type': NODE_LIT, 'value': v, 'op': 0,
                      'left': NO_CHILD, 'right': NO_CHILD})
        return my_id, v

    nodes.append(None)  # placeholder for recursion
    op = rng.choice([OP_ADD, OP_SUB, OP_MUL])   # always include mul

    if op == OP_MUL and force_leaf_mul:
        # force literal children
        left_id, left_v = _gen_subtree(0, rng, nodes, force_leaf_mul)
        right_id, right_v = _gen_subtree(0, rng, nodes, force_leaf_mul)
    else:
        left_id, left_v = _gen_subtree(depth_budget - 1, rng, nodes, force_leaf_mul)
        right_id, right_v = _gen_subtree(depth_budget - 1, rng, nodes, force_leaf_mul)

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


def has_nested_mul(nodes):
    """True if the tree contains `*` where at least one child is itself an op."""
    for n in nodes:
        if n is None:
            continue
        if n['type'] == NODE_OP and n['op'] == OP_MUL:
            lc = nodes[n['left']]
            rc = nodes[n['right']]
            if lc['type'] == NODE_OP or rc['type'] == NODE_OP:
                return True
    return False


def has_any_mul(nodes):
    return any(n['type'] == NODE_OP and n['op'] == OP_MUL for n in nodes if n is not None)


def generate_tree(depth, split, rng):
    """Generate a random tree meeting the split constraint.

    split='train':
        Trees where `*` appears only with literal children (leaf-level mul).
        Add/sub can appear at any depth. Trees may still contain mul.
    split='test_mul':
        Trees where `*` has at least one NON-leaf child. The model must
        generalize multiplication to compound subexpressions even though it
        only saw leaf-level mul in training.
    split='all':
        Any valid tree.
    """
    attempts = 0
    while attempts < 500:
        attempts += 1
        nodes = []
        # Force leaf-level mul in training; allow everything else in test.
        force_leaf_mul = (split == 'train')
        _gen_subtree(depth, rng, nodes, force_leaf_mul)
        if split == 'train':
            # Safety: reject anything that accidentally has nested mul
            if has_nested_mul(nodes):
                continue
            return nodes
        elif split == 'test_mul':
            if has_nested_mul(nodes):
                return nodes
            else:
                continue
        else:
            return nodes
    raise RuntimeError(f'failed to generate tree for split={split}')


def make_cruxmini_batch(batch_size, split, depth=3, device='cpu'):
    """Build a batch of trees. Returns a dict of tensors."""
    rng = random.Random(None)  # per-batch fresh
    cats = []       # (B, N): 0 = lit, 1 = op
    ops = []        # (B, N): 0-2 for ops, ignored for lits
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
# CruxMini V4-style Circuit
# ---------------------------------------------------------------------------

@dataclass
class CruxMiniConfig:
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


class CruxMiniCircuit(nn.Module):
    """V4-style circuit for arithmetic tree evaluation.

    Slot state per node = distribution over integer values (B, N, n_ints).
    Leaves are initialized to one-hot of their literal value. Op nodes
    update via a learned (n_ops, n_ints, n_ints, n_ints) lookup table.

    The lookup is the content-structure separator: the "structure"
    (op_type) indexes which lookup slice to use; the "content" (left and
    right integer distributions) is combined via outer product and
    element-wise multiply with the lookup slice, then summed to produce
    the output distribution.

    Tied projection: we could equivalently parameterize the circuit via a
    d_inner-dim embedding per int, but since we operate on distributions
    directly the tied embedding is effectively the identity projection
    over the n_ints discrete values. We DO learn a d_inner embedding for
    the readout in case a richer output space helps, but the core of the
    circuit is the integer-distribution lookup.
    """
    def __init__(self, cfg: CruxMiniConfig):
        super().__init__()
        self.cfg = cfg
        # Learned op lookup: (n_ops, n_ints, n_ints, n_ints)
        # op_table[op, i, j, k] = logit that "(i OP j) = k"
        # Small random init -- gives training something to move from without
        # baking in a direction that might contradict arithmetic.
        self.op_table = nn.Parameter(torch.randn(
            cfg.n_ops, cfg.n_ints, cfg.n_ints, cfg.n_ints
        ) * 0.1)

    def _init_state(self, cats, lits, mask):
        """Literals -> one-hot over ints. Non-literals -> uniform (will be overwritten)."""
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
        """One pass: for each node flagged in update_mask, compute the op result
        from children's current state.

        state:       (B, N, n_ints) -- current integer distributions
        cats, ops, left, right, mask, update_mask: (B, N)

        If return_logits is True, return (new_state, node_logits) where
        node_logits is the un-softmaxed per-node logit tensor for this pass
        (used for root readout to avoid numerical issues with log(softmax(...))).
        """
        B, N, n_ints = state.shape
        n_ops = self.cfg.n_ops
        device = state.device

        # Gather children's distributions
        l_exp = left.unsqueeze(-1).expand(-1, -1, n_ints)
        r_exp = right.unsqueeze(-1).expand(-1, -1, n_ints)
        left_dist = torch.gather(state, 1, l_exp)     # (B, N, n_ints)
        right_dist = torch.gather(state, 1, r_exp)    # (B, N, n_ints)

        # Soft lookup: for each node, pick the op_table slice based on ops[b, n]
        slice_table = self.op_table[ops.clamp(0, n_ops - 1)]   # (B, N, n_ints, n_ints, n_ints)

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
        device = batch['cats'].device

        cats = batch['cats']
        ops = batch['ops']
        lits = batch['lits']
        left = batch['left'].clamp(0, N - 1)
        right = batch['right'].clamp(0, N - 1)
        mask = batch['mask']

        state = self._init_state(cats, lits, mask)
        op_mask = (cats == NODE_OP) & mask

        # Propagate for all but the last pass using softmax distributions...
        for _ in range(cfg.n_passes - 1):
            state = self._op_step(state, cats, ops, left, right, mask, op_mask)
        # ...and return raw LOGITS from the final pass so the root loss uses
        # proper cross_entropy (better gradient flow than log(softmax(...))).
        state, node_logits = self._op_step(
            state, cats, ops, left, right, mask, op_mask, return_logits=True
        )

        # Root readout: use LOGITS for the root (node 0). Fall back to the
        # (constant) literal distribution if the root is a literal.
        op_node_logits = node_logits[:, 0, :]                 # (B, n_ints)
        root_is_lit = (cats[:, 0] == NODE_LIT).unsqueeze(-1).float()
        # For literal roots, the "logits" are just the one-hot scaled
        lit_logits = state[:, 0, :] * 10.0                    # saturate softmax
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
        b = make_cruxmini_batch(n_eval, split='train', device=device)
        results['acc_train'] = accuracy(model(b), b['targets'])
        b = make_cruxmini_batch(n_eval, split='test_mul', device=device)
        results['acc_test_mul'] = accuracy(model(b), b['targets'])
    model.train()
    return results


def train_cruxmini(model, cfg, device):
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

        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                tr_acc = accuracy(logits, batch['targets'])
            print(f"  [cruxmini] step {step:4d}/{cfg.n_steps}  loss={loss.item():.4f}  "
                  f"train_batch={tr_acc:.3f}  lr={lr:.2e}", flush=True)

        if step % cfg.eval_every == 0:
            ev = evaluate(model, cfg.n_eval, device)
            print(f"  [cruxmini] eval @ step {step}  train={ev['acc_train']:.3f}  "
                  f"test_mul={ev['acc_test_mul']:.3f}", flush=True)
            history.append({'step': step, **ev})

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--n-steps', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = CruxMiniConfig(
        lr=args.lr,
        n_steps=500 if args.smoke else args.n_steps,
        eval_every=100 if args.smoke else 250,
        n_eval=200 if args.smoke else 500,
    )

    print("=" * 66)
    print("CruxMini -- Nested arithmetic tree evaluation (V4-style circuit)")
    print("=" * 66)
    print(f"  Grammar: binary arithmetic trees, ops in {{+,-,*}}, ints in [0,9]")
    print(f"  Train: trees where * has LITERAL children only")
    print(f"  Test:  trees where * has at least one NON-literal child")
    print(f"  Arch:  tied integer distribution + learned op lookup table")
    print(f"  n_steps={cfg.n_steps}  batch={cfg.batch_size}  lr={cfg.lr}")
    print("=" * 66, flush=True)

    model = CruxMiniCircuit(cfg).to(device)
    p = count_params(model)
    print(f"\nCruxMini circuit: {p:,} params\n", flush=True)

    t0 = time.time()
    history = train_cruxmini(model, cfg, device)
    t_train = time.time() - t0

    final = evaluate(model, cfg.n_eval * 2, device)

    print("\n" + "=" * 66)
    print("FINAL RESULTS")
    print("=" * 66)
    print(f"  Train acc (leaf-level *):           {final['acc_train']:.3f}")
    print(f"  Test_mul acc (compound *):          {final['acc_test_mul']:.3f}")
    print(f"  Random baseline (10 classes):       0.100")
    print(f"  Training time:                      {t_train:.1f}s")
    print("=" * 66)

    out_tag = f'cruxmini_s{args.seed}'
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
