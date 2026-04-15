"""
Matched transformer baseline for CLUTRR V4.

Same data pipeline, same train loop, same per-hop eval as `clutrr_v4.py`.
The ONLY difference is the model: a standard decoder-only transformer that
receives the same (edge_src, edge_rel, edge_tgt, query_src, query_tgt) input
through an edge-encoder that builds one token per edge + one query token.

Why edge-as-token: V4 is a message-passing GNN over entity slots with edge
content `[src_state, rel, tgt_state]` as the per-edge message. The natural
matched baseline is a transformer whose tokens carry exactly the same info
(src_emb, rel_emb, tgt_emb concatenated then projected) plus a query token.
The transformer gets UNRESTRICTED bidirectional attention over the edge set.
Any win for V4 over this baseline is NOT explained by "the TF doesn't see
the right input" -- it sees the same thing, just through attention.

Matched-params target: V4 is ~272K params. TF at d=64, n_heads=4, n_layers=4
lands near 256K, within 6% of V4. TF slightly under-matched, which if
anything handicaps V4.

Multi-seed: --seed flag. Output dir is seed-specific:
`results/clutrr_tf_d{d}_L{L}_s{seed}/results.json`.

Usage:
    python train_clutrr_tf.py --smoke
    python train_clutrr_tf.py --epochs 25 --seed 42
    python train_clutrr_tf.py --epochs 25 --seed 1337
    python train_clutrr_tf.py --epochs 25 --seed 7
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse ALL data loading + training from clutrr_v4 so V4 and TF see identical
# data, identical batching, identical eval metrics.
from clutrr_v4 import (
    RELATIONS,
    N_REL,
    MAX_ENTITIES,
    MAX_EDGES,
    try_load_clutrr_hf,
    build_training_set,
    train,
)


# ---------------------------------------------------------------------------
# Matched transformer model
# ---------------------------------------------------------------------------

class CLUTRRTransformer(nn.Module):
    """Decoder-only transformer matched to CLUTRRV4.

    Input: the same (edge_src, edge_rel, edge_tgt, n_edges, query_src, query_tgt)
    that CLUTRRV4 takes. Each edge becomes ONE token built from
    concat[entity_embed(src), rel_embed(rel), entity_embed(tgt)] projected to
    d_model. Query becomes ONE token at the end. Padding edges are masked out
    via `src_key_padding_mask`.

    Readout: final hidden state of the query token -> classifier -> 20-way
    relation logits. Same 20-class head as CLUTRRV4.
    """

    def __init__(self, d_model=64, n_heads=4, n_layers=4, dropout=0.0):
        super().__init__()
        self.d_model = d_model

        self.entity_embed = nn.Embedding(MAX_ENTITIES, d_model)
        self.rel_embed = nn.Embedding(N_REL, d_model)

        # Edge encoder: [src_emb, rel_emb, tgt_emb] -> d_model
        self.edge_enc = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Query encoder: [q_src_emb, q_tgt_emb] -> d_model
        self.query_enc = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Learned positional encoding over token positions (edges + 1 query)
        self.pos_embed = nn.Embedding(MAX_EDGES + 1, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Same 20-class classifier head as CLUTRRV4 (for matched readout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, N_REL),
        )

    def forward(self, edge_src, edge_rel, edge_tgt, n_edges, query_src, query_tgt):
        B = edge_src.size(0)
        max_e = edge_src.size(1)
        device = edge_src.device

        # Build edge tokens
        src_emb = self.entity_embed(edge_src.clamp(0, MAX_ENTITIES - 1))
        rel_emb = self.rel_embed(edge_rel.clamp(0, N_REL - 1))
        tgt_emb = self.entity_embed(edge_tgt.clamp(0, MAX_ENTITIES - 1))
        edge_tokens = self.edge_enc(torch.cat([src_emb, rel_emb, tgt_emb], dim=-1))
        # edge_tokens: (B, max_e, d_model)

        # Build query token
        q_src_emb = self.entity_embed(query_src.clamp(0, MAX_ENTITIES - 1))
        q_tgt_emb = self.entity_embed(query_tgt.clamp(0, MAX_ENTITIES - 1))
        query_token = self.query_enc(torch.cat([q_src_emb, q_tgt_emb], dim=-1)).unsqueeze(1)
        # query_token: (B, 1, d_model)

        # Concatenate edges + query
        tokens = torch.cat([edge_tokens, query_token], dim=1)  # (B, max_e+1, d_model)

        # Add positional encoding
        positions = torch.arange(tokens.size(1), device=device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.pos_embed(positions)

        # Padding mask: True = padding (masked out), False = valid
        edge_pad_mask = torch.arange(max_e, device=device).unsqueeze(0) >= n_edges.unsqueeze(1)
        query_valid = torch.zeros(B, 1, dtype=torch.bool, device=device)
        src_key_padding_mask = torch.cat([edge_pad_mask, query_valid], dim=1)

        # Transformer: bidirectional attention over edges + query
        h = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)

        # Readout from query token (last position)
        query_final = h[:, -1, :]  # (B, d_model)
        return self.classifier(query_final)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 66)
    print("CLUTRR matched Transformer baseline")
    print("=" * 66)
    print(f"  device={device}  seed={args.seed}")
    print(f"  d_model={args.d_model}  n_heads={args.n_heads}  n_layers={args.n_layers}")
    print(f"  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    print("=" * 66, flush=True)

    print("Loading CLUTRR (reusing clutrr_v4 data pipeline)...")
    ds = try_load_clutrr_hf()
    if ds is None:
        print("FAILED to load CLUTRR.")
        return

    print("\nDataset keys:", list(ds.keys()))
    max_train = 100 if args.smoke else None
    max_test = 100 if args.smoke else None
    train_ex = build_training_set(ds, 'train', max_examples=max_train)
    test_ex = build_training_set(ds, 'test', max_examples=max_test) if 'test' in ds else \
              build_training_set(ds, 'validation', max_examples=max_test) if 'validation' in ds else train_ex[:200]

    if not train_ex:
        print("ERROR: no training examples.")
        return

    epochs = 3 if args.smoke else args.epochs
    model = CLUTRRTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nCLUTRR Transformer: {n_params:,} params")
    print(f"Training on {len(train_ex)} examples, testing on {len(test_ex)}...\n", flush=True)

    t0 = time.time()
    # Reuse the EXACT same train() from clutrr_v4 for fair comparison
    history = train(model, train_ex, test_ex, epochs, args.batch, args.lr, device)
    elapsed = time.time() - t0

    # Find best epoch by test_acc (same protocol as CLUTRR paper / clutrr_v4 reporting)
    best = max(history, key=lambda h: h['test_acc']) if history else None

    print("\n" + "=" * 66)
    print(f"DONE in {elapsed:.1f}s")
    if best:
        print(f"  best epoch:       {best['epoch']}")
        print(f"  best test_acc:    {best['test_acc']:.3f}")
        print(f"  best per-hop:")
        for t in sorted(best['per_task_acc'].keys()):
            print(f"    {t}: {best['per_task_acc'][t]:.3f}")
    print("=" * 66)

    out_dir = Path(f"results/clutrr_tf_d{args.d_model}_L{args.n_layers}_s{args.seed}")
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
