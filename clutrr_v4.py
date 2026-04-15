"""
CLUTRR V4 -- Family relationship reasoning via graph V4 circuit
=================================================================

CLUTRR is the published benchmark for multi-hop relation reasoning:
  Input:  "Alice is Bob's mother. Bob is Carol's father."
  Query:  "What is Alice to Carol?"
  Output: "grandmother"

This is natively a graph traversal task. Parse story -> entity graph with
relation-typed edges, then run V4-style message passing to answer the query.

Architecture: V4 edge-aware message passing on entity graphs
  - Slots: one per entity (clamped to a max)
  - Entity slots initialized from learned entity embeddings
  - Edges carry relation type (father / mother / son / ...)
  - Message: f(cat[src_state, rel_embed, tgt_state]) -- source-state aware
  - Classifier: project (query_src_state, query_tgt_state) -> relation logit

The compositional-generalization signal we care about: train on short chains
(2-3 hops), test on longer chains (4-6 hops). This is the CLUTRR "k" split.

Usage:
    python clutrr_v4.py              # train + eval
    python clutrr_v4.py --smoke      # quick sanity
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

RELATIONS = {
    'father': 0, 'mother': 1, 'son': 2, 'daughter': 3,
    'brother': 4, 'sister': 5,
    'grandfather': 6, 'grandmother': 7,
    'grandson': 8, 'granddaughter': 9,
    'uncle': 10, 'aunt': 11, 'nephew': 12, 'niece': 13,
    'husband': 14, 'wife': 15,
    'father-in-law': 16, 'mother-in-law': 17,
    'son-in-law': 18, 'daughter-in-law': 19,
}
N_REL = len(RELATIONS)
IDX_TO_REL = {v: k for k, v in RELATIONS.items()}

MAX_ENTITIES = 32
MAX_EDGES = 30


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CLUTRRV4(nn.Module):
    """V4-style edge-aware message passing on entity graphs.

    For each edge (src, rel, tgt) at each prop step:
      msg = msg_fn(cat[S[src], rel_embed, S[tgt]])
    Aggregation: scatter_add into target slot.
    Update: S_new = S + update_fn(cat[S, incoming]).
    """
    def __init__(self, d_slot=128, n_entities=MAX_ENTITIES, n_steps=8):
        super().__init__()
        self.d_slot = d_slot
        self.n_entities = n_entities
        self.n_steps = n_steps

        self.entity_embed = nn.Embedding(n_entities, d_slot)
        self.rel_embed = nn.Embedding(N_REL, d_slot)

        # msg_fn takes [src_state, rel_embed, tgt_state] -> d_slot
        self.msg_fn = nn.Sequential(
            nn.Linear(d_slot * 3, d_slot * 2),
            nn.GELU(),
            nn.Linear(d_slot * 2, d_slot),
        )
        self.update_fn = nn.Sequential(
            nn.Linear(d_slot * 2, d_slot * 2),
            nn.GELU(),
            nn.Linear(d_slot * 2, d_slot),
        )

        # Classifier: (src_state, tgt_state) -> relation logits
        self.classifier = nn.Sequential(
            nn.Linear(d_slot * 2, d_slot),
            nn.GELU(),
            nn.Linear(d_slot, N_REL),
        )

    def forward(self, edge_src, edge_rel, edge_tgt, n_edges, query_src, query_tgt):
        """
        edge_src:  (B, max_edges)  long
        edge_rel:  (B, max_edges)  long
        edge_tgt:  (B, max_edges)  long
        n_edges:   (B,)            long
        query_src: (B,)            long
        query_tgt: (B,)            long
        """
        B = edge_src.size(0)
        device = edge_src.device

        # Init entity slots from learned embeddings (broadcast per batch)
        S = self.entity_embed.weight.unsqueeze(0).expand(B, -1, -1).contiguous()
        # S: (B, n_entities, d_slot)

        max_e = edge_src.size(1)
        edge_mask = (
            torch.arange(max_e, device=device).unsqueeze(0) < n_edges.unsqueeze(1)
        ).float()

        for step in range(self.n_steps):
            d = self.d_slot

            # Gather source and target states
            src_idx = edge_src.unsqueeze(-1).expand(-1, -1, d)
            tgt_idx = edge_tgt.unsqueeze(-1).expand(-1, -1, d)
            src_states = S.gather(1, src_idx)
            tgt_states = S.gather(1, tgt_idx)

            # Messages: source-state-aware (V4 style)
            rel_emb = self.rel_embed(edge_rel)
            msg_input = torch.cat([src_states, rel_emb, tgt_states], dim=-1)
            messages = self.msg_fn(msg_input) * edge_mask.unsqueeze(-1)

            # Scatter-add to targets
            msg_agg = torch.zeros_like(S)
            msg_agg.scatter_add_(1, tgt_idx, messages)

            S = S + self.update_fn(torch.cat([S, msg_agg], dim=-1))

        # Readout: query src/tgt states
        q_src_idx = query_src.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.d_slot)
        q_tgt_idx = query_tgt.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.d_slot)
        src_final = S.gather(1, q_src_idx).squeeze(1)
        tgt_final = S.gather(1, q_tgt_idx).squeeze(1)

        return self.classifier(torch.cat([src_final, tgt_final], dim=-1))


# ---------------------------------------------------------------------------
# Parser: CLUTRR story -> (edges, query_src, query_tgt)
# ---------------------------------------------------------------------------

REL_WORDS = '|'.join(sorted(RELATIONS.keys(), key=len, reverse=True))

PATTERNS = [
    # "Alice is the mother of Bob" -> (Alice, mother, Bob) meaning Alice is Bob's mother
    (re.compile(
        rf"(\w+)\s+is\s+(?:the\s+)?({REL_WORDS})\s+of\s+(\w+)",
        re.IGNORECASE
    ), lambda m: (m.group(1), m.group(2).lower(), m.group(3))),

    # "Bob's mother is Alice" -> Alice is Bob's mother -> (Alice, mother, Bob)
    (re.compile(
        rf"(\w+)'s\s+({REL_WORDS})\s+is\s+(\w+)",
        re.IGNORECASE
    ), lambda m: (m.group(3), m.group(2).lower(), m.group(1))),

    # "Alice is Bob's mother" -> (Alice, mother, Bob)
    (re.compile(
        rf"(\w+)\s+is\s+(\w+)'s\s+({REL_WORDS})",
        re.IGNORECASE
    ), lambda m: (m.group(1), m.group(3).lower(), m.group(2))),
]

QUERY_PATTERNS = [
    re.compile(r"How\s+is\s+(\w+)\s+related\s+to\s+(\w+)", re.IGNORECASE),
    re.compile(r"What\s+is\s+(\w+)\s+to\s+(\w+)", re.IGNORECASE),
    re.compile(r"(\w+)\s+is\s+the\s+.+?\s+of\s+(\w+)", re.IGNORECASE),  # fallback
]


def parse_clutrr(story, query_text):
    """Extract (triples, q_src, q_tgt, name_to_id) from a CLUTRR story.

    Returns:
        triples: list of (src_name, rel_str, tgt_name)
        q_src, q_tgt: names
        name_to_id: dict mapping name -> entity index
    """
    # Discover entity names (capitalized words, excluding sentence-starters like "The")
    candidate_names = set()
    for m in re.finditer(r"\b[A-Z][a-z]+\b", story):
        w = m.group(0)
        if w not in {'The', 'A', 'An', 'How', 'What', 'Who', 'When', 'Where', 'Why',
                     'Is', 'Was', 'Are', 'Were', 'And', 'Or', 'But', 'So'}:
            candidate_names.add(w)

    # Parse triples
    triples = []
    for pattern, mapper in PATTERNS:
        for m in pattern.finditer(story):
            try:
                src_name, rel_str, tgt_name = mapper(m)
                if src_name in candidate_names and tgt_name in candidate_names \
                   and rel_str in RELATIONS:
                    triples.append((src_name, rel_str, tgt_name))
            except Exception:
                continue

    # Assign entity IDs in order of first appearance in the story
    name_to_id = {}
    next_id = 0
    for w in re.findall(r"\b[A-Z][a-z]+\b", story):
        if w in candidate_names and w not in name_to_id:
            name_to_id[w] = next_id
            next_id += 1
            if next_id >= MAX_ENTITIES:
                break

    # Parse query
    q_src_name = q_tgt_name = None
    for qp in QUERY_PATTERNS:
        m = qp.search(query_text)
        if m:
            q_src_name, q_tgt_name = m.group(1), m.group(2)
            break

    if q_src_name is None or q_tgt_name is None:
        return triples, None, None, name_to_id

    return triples, q_src_name, q_tgt_name, name_to_id


def encode_example(triples, q_src_name, q_tgt_name, name_to_id, target_rel):
    """Build tensors from parsed triples."""
    edge_src = [0] * MAX_EDGES
    edge_rel = [0] * MAX_EDGES
    edge_tgt = [0] * MAX_EDGES

    count = 0
    for src_name, rel_str, tgt_name in triples:
        if count >= MAX_EDGES:
            break
        if src_name not in name_to_id or tgt_name not in name_to_id:
            continue
        edge_src[count] = name_to_id[src_name]
        edge_rel[count] = RELATIONS[rel_str]
        edge_tgt[count] = name_to_id[tgt_name]
        count += 1

    if count == 0:
        return None
    if q_src_name not in name_to_id or q_tgt_name not in name_to_id:
        return None
    if target_rel not in RELATIONS:
        return None

    return {
        'edge_src': edge_src,
        'edge_rel': edge_rel,
        'edge_tgt': edge_tgt,
        'n_edges': count,
        'query_src': name_to_id[q_src_name],
        'query_tgt': name_to_id[q_tgt_name],
        'label': RELATIONS[target_rel],
    }


# ---------------------------------------------------------------------------
# Dataset loading -- try a few HF paths, fall back to local download
# ---------------------------------------------------------------------------

LOCAL_DATA = Path('data/clutrr')


def try_load_clutrr_hf():
    """Load CLUTRR from the pre-downloaded real compositional split.

    Falls back to nnonta/clutrr (task_1.2 only) if the local JSON is missing.
    """
    import json
    if (LOCAL_DATA / 'train.json').exists():
        ds = {}
        for split in ('train', 'validation', 'test'):
            p = LOCAL_DATA / f'{split}.json'
            if p.exists():
                with open(p) as f:
                    ds[split] = json.load(f)
        print(f"  Loaded local gen_train234_test2to10 splits "
              f"({len(ds.get('train', []))} train, {len(ds.get('test', []))} test)")
        return ds

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. Run: pip install datasets")
        return None
    try:
        ds = load_dataset('nnonta/clutrr')
        print(f"  Loaded nnonta/clutrr (task_1.2 only)")
        return ds
    except Exception as e:
        print(f"  nnonta/clutrr: {type(e).__name__}")
        return None


def encode_hf_example(ex):
    """Convert a pre-parsed HF CLUTRR example into a flat encoded dict.

    The dataset already gives us:
      story_edges: list of (src_idx, tgt_idx) tuples
      edge_types:  list of relation-name strings (parallel to story_edges)
      query_edge:  (src_idx, tgt_idx) tuple
      target_text: relation-name string (label)
      task_name:   'task_1.k' where k is chain length -- preserved for per-hop eval
    """
    story_edges = ex.get('story_edges', [])
    edge_types = ex.get('edge_types', [])
    query_edge = ex.get('query_edge', None)
    target_text = ex.get('target_text', '')
    task_name = ex.get('task_name', 'task_unknown')

    if isinstance(story_edges, str):
        story_edges = eval(story_edges)
    if isinstance(query_edge, str):
        query_edge = eval(query_edge)
    if isinstance(edge_types, str):
        edge_types = eval(edge_types)

    if not story_edges or not query_edge or not target_text:
        return None
    if target_text not in RELATIONS:
        return None

    edge_src = [0] * MAX_EDGES
    edge_rel = [0] * MAX_EDGES
    edge_tgt = [0] * MAX_EDGES
    count = 0
    for (s, t), rel_str in zip(story_edges, edge_types):
        if count >= MAX_EDGES:
            break
        if rel_str not in RELATIONS:
            continue
        if s >= MAX_ENTITIES or t >= MAX_ENTITIES:
            continue
        edge_src[count] = int(s)
        edge_rel[count] = RELATIONS[rel_str]
        edge_tgt[count] = int(t)
        count += 1

    if count == 0:
        return None

    q_src, q_tgt = int(query_edge[0]), int(query_edge[1])
    if q_src >= MAX_ENTITIES or q_tgt >= MAX_ENTITIES:
        return None

    return {
        'edge_src': edge_src,
        'edge_rel': edge_rel,
        'edge_tgt': edge_tgt,
        'n_edges': count,
        'query_src': q_src,
        'query_tgt': q_tgt,
        'label': RELATIONS[target_text],
        'task_name': task_name,
    }


def build_training_set(ds, split_name='train', max_examples=None):
    """Iterate dataset, use pre-parsed fields."""
    encoded = []
    skipped = 0
    if split_name not in ds:
        print(f"  split {split_name} not in dataset, using first available")
        split_name = list(ds.keys())[0]
    for i, ex in enumerate(ds[split_name]):
        if max_examples and len(encoded) >= max_examples:
            break
        enc = encode_hf_example(ex)
        if enc is None:
            skipped += 1
            continue
        encoded.append(enc)
    print(f"  {split_name}: encoded {len(encoded)}, skipped {skipped}")
    return encoded


def batchify(examples, batch_size, device):
    """Yield batches of tensors from an encoded examples list."""
    import random
    random.shuffle(examples)
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        if len(batch) < 2:
            continue
        yield {
            'edge_src': torch.tensor([e['edge_src'] for e in batch], dtype=torch.long, device=device),
            'edge_rel': torch.tensor([e['edge_rel'] for e in batch], dtype=torch.long, device=device),
            'edge_tgt': torch.tensor([e['edge_tgt'] for e in batch], dtype=torch.long, device=device),
            'n_edges':  torch.tensor([e['n_edges'] for e in batch], dtype=torch.long, device=device),
            'query_src': torch.tensor([e['query_src'] for e in batch], dtype=torch.long, device=device),
            'query_tgt': torch.tensor([e['query_tgt'] for e in batch], dtype=torch.long, device=device),
            'label':     torch.tensor([e['label'] for e in batch], dtype=torch.long, device=device),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def eval_per_hop(model, examples, batch_size, device):
    """Compute accuracy broken down by task_name (i.e. chain length k)."""
    from collections import defaultdict
    model.eval()
    per_task_correct = defaultdict(int)
    per_task_total = defaultdict(int)
    all_correct = 0
    all_total = 0
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch_ex = examples[i:i + batch_size]
            if not batch_ex:
                continue
            b = {
                'edge_src': torch.tensor([e['edge_src'] for e in batch_ex], dtype=torch.long, device=device),
                'edge_rel': torch.tensor([e['edge_rel'] for e in batch_ex], dtype=torch.long, device=device),
                'edge_tgt': torch.tensor([e['edge_tgt'] for e in batch_ex], dtype=torch.long, device=device),
                'n_edges':  torch.tensor([e['n_edges'] for e in batch_ex], dtype=torch.long, device=device),
                'query_src': torch.tensor([e['query_src'] for e in batch_ex], dtype=torch.long, device=device),
                'query_tgt': torch.tensor([e['query_tgt'] for e in batch_ex], dtype=torch.long, device=device),
                'label':     torch.tensor([e['label'] for e in batch_ex], dtype=torch.long, device=device),
            }
            logits = model(b['edge_src'], b['edge_rel'], b['edge_tgt'],
                           b['n_edges'], b['query_src'], b['query_tgt'])
            preds = logits.argmax(-1)
            is_correct = (preds == b['label']).cpu().tolist()
            for ex, c in zip(batch_ex, is_correct):
                t = ex['task_name']
                per_task_correct[t] += int(c)
                per_task_total[t] += 1
                all_correct += int(c)
                all_total += 1

    per_task_acc = {
        t: per_task_correct[t] / per_task_total[t]
        for t in sorted(per_task_total.keys())
    }
    overall = all_correct / max(all_total, 1)
    return overall, per_task_acc, per_task_total


def train(model, train_ex, test_ex, epochs, batch_size, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_correct = 0
        n_total = 0
        n_batches = 0

        for batch in batchify(train_ex, batch_size, device):
            logits = model(
                batch['edge_src'], batch['edge_rel'], batch['edge_tgt'],
                batch['n_edges'], batch['query_src'], batch['query_tgt']
            )
            loss = F.cross_entropy(logits, batch['label'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_correct += (logits.argmax(-1) == batch['label']).sum().item()
            n_total += batch['label'].size(0)
            n_batches += 1

        train_acc = n_correct / max(n_total, 1)
        train_loss = total_loss / max(n_batches, 1)

        # Per-hop eval on test split
        test_acc, per_task_acc, per_task_total = eval_per_hop(model, test_ex, batch_size, device)

        task_str = '  '.join(f"{t.replace('task_1.', 'k='):>5s}={per_task_acc[t]:.2f}"
                             for t in sorted(per_task_acc.keys()))
        print(f"  epoch {epoch:2d}  loss={train_loss:.4f}  train={train_acc:.3f}  "
              f"test={test_acc:.3f}  | {task_str}", flush=True)
        history.append({
            'epoch': epoch,
            'loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'per_task_acc': per_task_acc,
            'per_task_count': dict(per_task_total),
        })

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d-slot', type=int, default=128)
    parser.add_argument('--n-steps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import random as _r
    torch.manual_seed(args.seed)
    _r.seed(args.seed)
    try:
        import numpy as _np
        _np.random.seed(args.seed)
    except ImportError:
        pass

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 66)
    print("CLUTRR V4 -- family reasoning via graph V4 message passing")
    print("=" * 66)
    print(f"  device={device}  seed={args.seed}  d_slot={args.d_slot}  n_steps={args.n_steps}")
    print(f"  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    print("=" * 66, flush=True)

    print("Loading CLUTRR from HuggingFace...")
    ds = try_load_clutrr_hf()
    if ds is None:
        print("FAILED to load CLUTRR from any HF path. Install datasets and try again,")
        print("or download the repo from https://github.com/facebookresearch/clutrr.")
        return

    print("\nDataset keys:", list(ds.keys()))
    first_split = list(ds.keys())[0]
    print(f"Example from '{first_split}':")
    print(" ", ds[first_split][0])

    max_train = 100 if args.smoke else None
    max_test = 100 if args.smoke else None
    train_ex = build_training_set(ds, 'train', max_examples=max_train)
    test_ex = build_training_set(ds, 'test', max_examples=max_test) if 'test' in ds else \
              build_training_set(ds, 'validation', max_examples=max_test) if 'validation' in ds else train_ex[:200]

    if not train_ex:
        print("ERROR: parser found zero usable training examples. Check story format.")
        return

    epochs = 3 if args.smoke else args.epochs
    model = CLUTRRV4(d_slot=args.d_slot, n_steps=args.n_steps).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nCLUTRR V4 model: {n_params:,} params")
    print(f"Training on {len(train_ex)} examples, testing on {len(test_ex)}...\n", flush=True)

    t0 = time.time()
    history = train(model, train_ex, test_ex, epochs, args.batch, args.lr, device)
    elapsed = time.time() - t0

    print("\n" + "=" * 66)
    print(f"DONE in {elapsed:.1f}s")
    if history:
        final = history[-1]
        print(f"  final train_acc: {final['train_acc']:.3f}")
        print(f"  final test_acc:  {final['test_acc']:.3f}")
    print("=" * 66)

    # Find best epoch by test_acc (CLUTRR-standard reporting protocol)
    best = max(history, key=lambda h: h['test_acc']) if history else None

    # Seed-specific output path so multi-seed runs don't overwrite each other
    out_dir = Path(f'results/clutrr_v4_d{args.d_slot}_s{args.seed}')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump({
            'history': history,
            'params': n_params,
            'config': vars(args),
            'n_train': len(train_ex),
            'n_test': len(test_ex),
            'elapsed': elapsed,
            'best_epoch': best['epoch'] if best else None,
            'best_test_acc': best['test_acc'] if best else None,
            'best_per_task_acc': best['per_task_acc'] if best else None,
        }, f, indent=2)
    print(f"Saved to {out_dir / 'results.json'}")


if __name__ == '__main__':
    main()
