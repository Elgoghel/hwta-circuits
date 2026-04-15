"""
Hierarchical WTA: Stable Discrete Routing at Scale
=====================================================
THE fix. Never more than 16-way competition, no matter the scale.

Level 1: Which group? (16 groups, 16-way WTA) - proven stable
Level 2: Which slot in group? (16 slots, 16-way WTA) - proven stable
Total capacity: 256 slots. Competition: always 16-way.

Scale: 16^2=256, 16^3=4096, 16^4=65536, 16^5=1M+

Matched params test: 1.7M hierarchical circuit vs 1.7M transformer.
If training is stable AND beats transformer: circuits scale.

Usage:
    python hierarchical_wta.py              # matched params head-to-head
    python hierarchical_wta.py --smoke      # quick test
"""

import argparse
import math
import time
import random
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaling_laws import (
    ScalableTransformer, make_graph_batch, count_params,
)


# ---------------------------------------------------------------------------
# Hierarchical WTA Circuit
# ---------------------------------------------------------------------------

class HierarchicalWTABlock(nn.Module):
    """Single propagation block with hierarchical WTA routing.

    Slots organized into groups. Competition is LOCAL:
    - Level 1: which group (n_groups-way WTA)
    - Level 2: which slot in group (group_size-way WTA)
    Never more than max(n_groups, group_size)-way competition.
    """

    def __init__(self, n_groups, group_size, d_slot):
        super().__init__()
        self.n_groups = n_groups
        self.group_size = group_size
        self.n_slots = n_groups * group_size
        self.d_slot = d_slot

        # Group-level keys (for level 1 routing)
        self.group_proj = nn.Linear(d_slot, n_groups, bias=False)

        # Within-group slot keys (for level 2 routing)
        self.slot_proj = nn.Linear(d_slot, group_size, bias=False)

        # Message function
        self.msg_fn = nn.Sequential(
            nn.Linear(d_slot, d_slot * 2), nn.GELU(), nn.Linear(d_slot * 2, d_slot))

        # Gate
        self.gate = nn.Sequential(
            nn.Linear(d_slot, d_slot), nn.GELU(), nn.Linear(d_slot, 1))

        # Update
        self.update = nn.Sequential(
            nn.Linear(d_slot * 2, d_slot * 2), nn.GELU(), nn.Linear(d_slot * 2, d_slot))

        self.ln = nn.LayerNorm(d_slot)

    def _hierarchical_route(self, X, S, tau=0.1):
        """Route tokens to slots via 2-level hierarchical WTA.

        X: (B, L, d) token representations
        S: (B, N_slots, d) slot states

        Returns: (B, N_slots, d) incoming messages per slot
        """
        B, L, d = X.shape
        N = self.n_slots
        G = self.n_groups
        K = self.group_size
        device = X.device

        # Compute messages and gates once
        msg = self.msg_fn(X)  # (B, L, d)
        gate = torch.sigmoid(self.gate(X))  # (B, L, 1)
        gated_msg = msg * gate  # (B, L, d)

        # Reshape slots into groups: (B, G, K, d)
        S_grouped = S.view(B, G, K, d)
        # Group centroids: (B, G, d)
        group_centroids = S_grouped.mean(dim=2)

        # Level 1: Which group? (G-way WTA per token)
        # Project tokens to group scores
        group_scores = self.group_proj(X)  # (B, L, G)

        if self.training:
            group_assign = F.gumbel_softmax(group_scores, tau=tau, hard=True, dim=-1)  # (B, L, G)
        else:
            group_idx = group_scores.argmax(dim=-1)  # (B, L)
            group_assign = F.one_hot(group_idx, G).float()  # (B, L, G)

        # Level 2: Which slot within group? (K-way WTA per token)
        slot_scores = self.slot_proj(X)  # (B, L, K)

        if self.training:
            slot_assign = F.gumbel_softmax(slot_scores, tau=tau, hard=True, dim=-1)  # (B, L, K)
        else:
            slot_idx = slot_scores.argmax(dim=-1)  # (B, L)
            slot_assign = F.one_hot(slot_idx, K).float()  # (B, L, K)

        # Combine: route message to the specific slot
        # full_assign: (B, L, G, K) -> (B, L, N)
        full_assign = group_assign.unsqueeze(-1) * slot_assign.unsqueeze(-2)  # (B, L, G, K)
        full_assign = full_assign.view(B, L, N)  # (B, L, N)

        # Scatter messages: (B, L, d) weighted by (B, L, N) -> (B, N, d)
        # incoming[b, n] = sum_l(full_assign[b, l, n] * gated_msg[b, l])
        incoming = torch.bmm(full_assign.transpose(1, 2), gated_msg)  # (B, N, d)

        return incoming

    def forward(self, X, S, tau=0.1):
        """
        X: (B, L, d) input tokens
        S: (B, N, d) slot states
        Returns: updated S
        """
        incoming = self._hierarchical_route(X, S, tau)

        # Update slots
        S = S + self.update(torch.cat([S, incoming], dim=-1))
        S = self.ln(S)

        return S


class HierarchicalCircuit(nn.Module):
    """Full hierarchical WTA circuit for graph traversal."""

    def __init__(self, n_groups=16, group_size=16, d_slot=256,
                 n_steps=32, n_relations=2, max_nodes=50):
        super().__init__()
        self.n_groups = n_groups
        self.group_size = group_size
        self.n_slots = n_groups * group_size
        self.d_slot = d_slot
        self.n_steps = n_steps

        # Input encoding
        self.node_embed = nn.Embedding(max_nodes, d_slot)
        self.rel_embed = nn.Embedding(n_relations, d_slot)
        self.edge_enc = nn.Sequential(
            nn.Linear(3 * d_slot, d_slot), nn.GELU(), nn.Linear(d_slot, d_slot))
        self.query_enc = nn.Sequential(
            nn.Linear(2 * d_slot, d_slot), nn.GELU(), nn.Linear(d_slot, d_slot))

        # Hierarchical routing blocks (shared weights across steps for efficiency)
        self.block = HierarchicalWTABlock(n_groups, group_size, d_slot)

        # Slot initialization
        self.S_init = nn.Parameter(torch.randn(n_groups * group_size, d_slot) * 0.02)

        # Readout
        self.head = nn.Sequential(
            nn.Linear(d_slot * 2, d_slot), nn.GELU(), nn.Linear(d_slot, 2))

        self.register_buffer('tau', torch.tensor(0.5))

    def forward(self, src, rel, tgt, mask, qsrc, qtgt):
        B = src.shape[0]
        N = self.n_slots
        d = self.d_slot
        device = src.device

        # Encode edges
        src_emb = self.node_embed(src.clamp(0, 49))
        tgt_emb = self.node_embed(tgt.clamp(0, 49))
        rel_emb = self.rel_embed(rel)
        X = self.edge_enc(torch.cat([src_emb, rel_emb, tgt_emb], dim=-1))

        # Query
        q = self.query_enc(torch.cat([
            self.node_embed(qsrc.clamp(0, 49)),
            self.node_embed(qtgt.clamp(0, 49))], dim=-1))
        X = torch.cat([X, q.unsqueeze(1)], dim=1)  # (B, L, d)

        # Init slots
        S = self.S_init.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Propagate with hierarchical WTA (gradient checkpointing for memory)
        use_checkpoint = self.training and S.numel() * 4 > 100_000_000  # >100MB activations
        tau_val = self.tau.item()  # lift sync out of the loop (was called per step)
        for step in range(self.n_steps):
            if use_checkpoint:
                S = torch.utils.checkpoint.checkpoint(
                    self.block, X, S, tau_val, use_reentrant=False)
            else:
                S = self.block(X, S, tau=tau_val)

        # Readout
        s_src = torch.gather(S, 1, qsrc.clamp(0, N-1).unsqueeze(1).unsqueeze(2).expand(-1, -1, d)).squeeze(1)
        s_tgt = torch.gather(S, 1, qtgt.clamp(0, N-1).unsqueeze(1).unsqueeze(2).expand(-1, -1, d)).squeeze(1)
        logits = self.head(torch.cat([s_src, s_tgt], dim=-1))
        return logits


# ---------------------------------------------------------------------------
# V2: Slot-to-Slot Attention + Attention Readout
# ---------------------------------------------------------------------------
# V1 problem: slots are isolated. Update function is per-slot, routing depends
# only on X, readout is positional gather. Wider d_slot overfits training
# depths; more slots waste capacity (workspace cannot reach the head).
#
# V2 fixes:
#   1. Slot self-attention inside the block — slots can communicate, so
#      workspace slots become useful and multi-hop traversal becomes real.
#   2. Attention readout — query node embedding attends over ALL slots,
#      not just slot index = node id. All slot capacity contributes.


class HierarchicalWTABlockV2(nn.Module):
    """Block with slot-to-slot self-attention added before the update.

    enable_slot_attn=False gives attention-readout-only (V2 circuit with V1 block).
    """

    def __init__(self, n_groups, group_size, d_slot, n_heads=8, enable_slot_attn=True):
        super().__init__()
        self.n_groups = n_groups
        self.group_size = group_size
        self.n_slots = n_groups * group_size
        self.d_slot = d_slot
        self.enable_slot_attn = enable_slot_attn

        # Hierarchical routing (unchanged from V1)
        self.group_proj = nn.Linear(d_slot, n_groups, bias=False)
        self.slot_proj = nn.Linear(d_slot, group_size, bias=False)

        self.msg_fn = nn.Sequential(
            nn.Linear(d_slot, d_slot * 2), nn.GELU(), nn.Linear(d_slot * 2, d_slot))
        self.gate = nn.Sequential(
            nn.Linear(d_slot, d_slot), nn.GELU(), nn.Linear(d_slot, 1))

        # Slot-to-slot self-attention (optional)
        if enable_slot_attn:
            self.slot_attn = nn.MultiheadAttention(
                d_slot, num_heads=n_heads, batch_first=True, bias=True)
            self.attn_ln = nn.LayerNorm(d_slot)

        # Update (unchanged from V1)
        self.update = nn.Sequential(
            nn.Linear(d_slot * 2, d_slot * 2), nn.GELU(), nn.Linear(d_slot * 2, d_slot))
        self.ln = nn.LayerNorm(d_slot)

    def _hierarchical_route(self, X, S, tau):
        B, L, d = X.shape
        N = self.n_slots
        G = self.n_groups
        K = self.group_size

        msg = self.msg_fn(X)
        gate = torch.sigmoid(self.gate(X))
        gated_msg = msg * gate

        group_scores = self.group_proj(X)
        slot_scores = self.slot_proj(X)

        if self.training:
            group_assign = F.gumbel_softmax(group_scores, tau=tau, hard=True, dim=-1)
            slot_assign = F.gumbel_softmax(slot_scores, tau=tau, hard=True, dim=-1)
        else:
            group_idx = group_scores.argmax(dim=-1)
            slot_idx = slot_scores.argmax(dim=-1)
            group_assign = F.one_hot(group_idx, G).float()
            slot_assign = F.one_hot(slot_idx, K).float()

        full_assign = group_assign.unsqueeze(-1) * slot_assign.unsqueeze(-2)
        full_assign = full_assign.view(B, L, N)

        incoming = torch.bmm(full_assign.transpose(1, 2), gated_msg)
        return incoming

    def forward(self, X, S, tau=0.1):
        incoming = self._hierarchical_route(X, S, tau)

        # Slot-to-slot self-attention (optional) -- enables multi-hop
        if self.enable_slot_attn:
            attn_out, _ = self.slot_attn(S, S, S, need_weights=False)
            S = self.attn_ln(S + attn_out)

        # Update slots
        S = S + self.update(torch.cat([S, incoming], dim=-1))
        S = self.ln(S)
        return S


class HierarchicalCircuitV2(nn.Module):
    """Graph-traversal circuit with slot-to-slot attention and attention readout.

    Architectural changes vs V1:
      - Block includes slot self-attention (HierarchicalWTABlockV2)
      - Readout: cross-attention from query node embedding to ALL slots
        (replaces positional gather which only saw slots 0-49)
      - No gradient checkpointing (torch.compile hates it; bf16+compile is
        enough)
    """

    def __init__(self, n_groups=16, group_size=16, d_slot=256,
                 n_steps=24, n_relations=2, max_nodes=50, n_heads=8,
                 enable_slot_attn=True):
        super().__init__()
        self.n_groups = n_groups
        self.group_size = group_size
        self.n_slots = n_groups * group_size
        self.d_slot = d_slot
        self.n_steps = n_steps

        # Input encoding
        self.node_embed = nn.Embedding(max_nodes, d_slot)
        self.rel_embed = nn.Embedding(n_relations, d_slot)
        self.edge_enc = nn.Sequential(
            nn.Linear(3 * d_slot, d_slot), nn.GELU(), nn.Linear(d_slot, d_slot))
        self.query_enc = nn.Sequential(
            nn.Linear(2 * d_slot, d_slot), nn.GELU(), nn.Linear(d_slot, d_slot))

        # Shared routing block with slot self-attention (optional)
        self.block = HierarchicalWTABlockV2(
            n_groups, group_size, d_slot, n_heads=n_heads,
            enable_slot_attn=enable_slot_attn)

        # Slot initialization
        self.S_init = nn.Parameter(torch.randn(n_groups * group_size, d_slot) * 0.02)

        # NEW: attention readout -- query projects, scores against all slots,
        # returns weighted sum. Every slot participates.
        self.readout_q_proj = nn.Linear(d_slot, d_slot, bias=False)

        # Head reads [attn_src, attn_tgt]
        self.head = nn.Sequential(
            nn.Linear(d_slot * 2, d_slot), nn.GELU(), nn.Linear(d_slot, 2))

        self.register_buffer('tau', torch.tensor(0.5))

    def _attn_readout(self, query_emb, S):
        """Cross-attention readout: query → all slots.

        query_emb: (B, d)
        S:         (B, N, d)
        Returns:   (B, d) attended slot state
        """
        q = self.readout_q_proj(query_emb)                          # (B, d)
        scale = 1.0 / (self.d_slot ** 0.5)
        scores = torch.einsum('bd,bnd->bn', q, S) * scale           # (B, N)
        attn = F.softmax(scores, dim=-1)                            # (B, N)
        out = torch.einsum('bn,bnd->bd', attn, S)                   # (B, d)
        return out

    def forward(self, src, rel, tgt, mask, qsrc, qtgt):
        B = src.shape[0]
        d = self.d_slot

        # Encode edges
        src_emb = self.node_embed(src.clamp(0, 49))
        tgt_emb = self.node_embed(tgt.clamp(0, 49))
        rel_emb = self.rel_embed(rel)
        X = self.edge_enc(torch.cat([src_emb, rel_emb, tgt_emb], dim=-1))

        # Query token
        q_tok = self.query_enc(torch.cat([
            self.node_embed(qsrc.clamp(0, 49)),
            self.node_embed(qtgt.clamp(0, 49))], dim=-1))
        X = torch.cat([X, q_tok.unsqueeze(1)], dim=1)

        # Init slots
        S = self.S_init.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Propagate (no checkpoint; compile will fuse)
        tau_val = self.tau.item()
        for step in range(self.n_steps):
            S = self.block(X, S, tau=tau_val)

        # Attention readout over ALL slots
        q_src_emb = self.node_embed(qsrc.clamp(0, 49))
        q_tgt_emb = self.node_embed(qtgt.clamp(0, 49))
        s_src = self._attn_readout(q_src_emb, S)
        s_tgt = self._attn_readout(q_tgt_emb, S)
        logits = self.head(torch.cat([s_src, s_tgt], dim=-1))
        return logits


# ---------------------------------------------------------------------------
# V3: Dynamic (state-dependent) routing -- the real fix
# ---------------------------------------------------------------------------
# V1 problem: msg_fn, gate, and routing all only depend on X (static edge
# tokens). Over 24 prop steps, the SAME messages are routed to the SAME slots
# every step. That isn't multi-hop -- it's a fixed-point iteration on each
# slot independently. Slots never talk, information never flows between them,
# and depth-30 hits a ceiling around 68.9%.
#
# V3 fix: concatenate the current slot-state summary to every token before
# computing messages + routing. As slots fill up, the messages adapt and the
# routing redirects. Information flows slot-to-slot via ROUTING ADAPTATION.
# Real multi-hop emerges from 24 steps of state-dependent routing.
#
# Keep V1's positional-gather readout (stable, no train/eval softmax drift).


class HierarchicalWTABlockV3(nn.Module):
    """Block with state-dependent routing and messages.

    CRITICAL INIT TRICK: the state-dependent half of all input projections
    is zero-initialized. This makes V3 START as V1 behaviorally (ignoring
    the state component) and gradually learn to use state via gradient
    descent. This avoids the cold-start problem where noisy S at init
    corrupts routing decisions.
    """

    def __init__(self, n_groups, group_size, d_slot):
        super().__init__()
        self.n_groups = n_groups
        self.group_size = group_size
        self.n_slots = n_groups * group_size
        self.d_slot = d_slot

        # All routing/messages now take [token, slot_summary] -> 2d input
        self.group_proj = nn.Linear(2 * d_slot, n_groups, bias=False)
        self.slot_proj = nn.Linear(2 * d_slot, group_size, bias=False)

        self.msg_fn = nn.Sequential(
            nn.Linear(2 * d_slot, d_slot * 2), nn.GELU(),
            nn.Linear(d_slot * 2, d_slot))
        self.gate = nn.Sequential(
            nn.Linear(2 * d_slot, d_slot), nn.GELU(),
            nn.Linear(d_slot, 1))

        # Update unchanged: it already sees S
        self.update = nn.Sequential(
            nn.Linear(d_slot * 2, d_slot * 2), nn.GELU(),
            nn.Linear(d_slot * 2, d_slot))
        self.ln = nn.LayerNorm(d_slot)

        # Residual-init: zero the state half of all input projections.
        # Start as V1-behavior; grow dynamic routing on top.
        with torch.no_grad():
            self.group_proj.weight[:, d_slot:].zero_()
            self.slot_proj.weight[:, d_slot:].zero_()
            self.msg_fn[0].weight[:, d_slot:].zero_()
            self.gate[0].weight[:, d_slot:].zero_()

    def forward(self, X, S, tau=0.1):
        B, L, d = X.shape
        N = self.n_slots
        G = self.n_groups
        K = self.group_size

        # DYNAMIC: every token now sees the current slot-state summary.
        # As S evolves across prop steps, msg_fn/router outputs change.
        # CRITICAL: detach the state summary from the gradient graph.
        # Gradients still flow through the slot update path; they just
        # don't flow BACKWARD through the routing decision into past slot
        # states. This prevents the 2N-deep gradient path that kills
        # optimization when prop_steps is large.
        S_summary = S.detach().mean(dim=1, keepdim=True).expand(-1, L, -1)   # (B, L, d)
        XS = torch.cat([X, S_summary], dim=-1)                                # (B, L, 2d)

        msg = self.msg_fn(XS)                                         # (B, L, d)
        gate = torch.sigmoid(self.gate(XS))                           # (B, L, 1)
        gated_msg = msg * gate                                        # (B, L, d)

        group_scores = self.group_proj(XS)                            # (B, L, G)
        slot_scores = self.slot_proj(XS)                              # (B, L, K)

        if self.training:
            group_assign = F.gumbel_softmax(group_scores, tau=tau, hard=True, dim=-1)
            slot_assign = F.gumbel_softmax(slot_scores, tau=tau, hard=True, dim=-1)
        else:
            group_idx = group_scores.argmax(dim=-1)
            slot_idx = slot_scores.argmax(dim=-1)
            group_assign = F.one_hot(group_idx, G).float()
            slot_assign = F.one_hot(slot_idx, K).float()

        full_assign = group_assign.unsqueeze(-1) * slot_assign.unsqueeze(-2)
        full_assign = full_assign.view(B, L, N)

        incoming = torch.bmm(full_assign.transpose(1, 2), gated_msg)

        # Slot update (unchanged)
        S = S + self.update(torch.cat([S, incoming], dim=-1))
        S = self.ln(S)
        return S


class HierarchicalCircuitV3(nn.Module):
    """Graph-traversal circuit with dynamic routing block + V1 gather readout."""

    def __init__(self, n_groups=16, group_size=16, d_slot=256,
                 n_steps=24, n_relations=2, max_nodes=50):
        super().__init__()
        self.n_groups = n_groups
        self.group_size = group_size
        self.n_slots = n_groups * group_size
        self.d_slot = d_slot
        self.n_steps = n_steps

        self.node_embed = nn.Embedding(max_nodes, d_slot)
        self.rel_embed = nn.Embedding(n_relations, d_slot)
        self.edge_enc = nn.Sequential(
            nn.Linear(3 * d_slot, d_slot), nn.GELU(), nn.Linear(d_slot, d_slot))
        self.query_enc = nn.Sequential(
            nn.Linear(2 * d_slot, d_slot), nn.GELU(), nn.Linear(d_slot, d_slot))

        self.block = HierarchicalWTABlockV3(n_groups, group_size, d_slot)

        self.S_init = nn.Parameter(torch.randn(n_groups * group_size, d_slot) * 0.02)

        # V1 gather readout -- robust, no softmax train/eval drift
        self.head = nn.Sequential(
            nn.Linear(d_slot * 2, d_slot), nn.GELU(), nn.Linear(d_slot, 2))

        self.register_buffer('tau', torch.tensor(0.5))

    def forward(self, src, rel, tgt, mask, qsrc, qtgt):
        B = src.shape[0]
        N = self.n_slots
        d = self.d_slot

        src_emb = self.node_embed(src.clamp(0, 49))
        tgt_emb = self.node_embed(tgt.clamp(0, 49))
        rel_emb = self.rel_embed(rel)
        X = self.edge_enc(torch.cat([src_emb, rel_emb, tgt_emb], dim=-1))

        q = self.query_enc(torch.cat([
            self.node_embed(qsrc.clamp(0, 49)),
            self.node_embed(qtgt.clamp(0, 49))], dim=-1))
        X = torch.cat([X, q.unsqueeze(1)], dim=1)

        S = self.S_init.unsqueeze(0).expand(B, -1, -1).contiguous()

        tau_val = self.tau.item()
        for step in range(self.n_steps):
            S = self.block(X, S, tau=tau_val)

        # Gather readout (stable)
        s_src = torch.gather(S, 1, qsrc.clamp(0, N-1).unsqueeze(1).unsqueeze(2).expand(-1, -1, d)).squeeze(1)
        s_tgt = torch.gather(S, 1, qtgt.clamp(0, N-1).unsqueeze(1).unsqueeze(2).expand(-1, -1, d)).squeeze(1)
        logits = self.head(torch.cat([s_src, s_tgt], dim=-1))
        return logits


# ---------------------------------------------------------------------------
# V4: Edge-aware message passing (real GNN multi-hop)
# ---------------------------------------------------------------------------
# V1 problem: messages come from edge tokens X and never carry slot state.
#   The 24 prop steps are 24 iterations on accumulating static messages -- no
#   slot-to-slot information flow.
#
# V3 problem: S.mean() summary is a near-constant vector; router barely sees
#   any useful state signal. Dynamic routing degenerates.
#
# V4 fix: for each edge (src, rel, tgt), the MESSAGE carries S[src] -- the
#   current state of the SOURCE SLOT. As slots fill up across prop steps, the
#   messages evolve: step 1 slot A starts empty; step 5 slot A contains "I'm
#   reachable from origin"; step 10 edge (A,B)'s message now carries that info
#   to slot B. This is the standard GNN source-state message passing, just
#   with hierarchical WTA routing replacing sum/mean aggregation at the target.
#
# Keep V1 gather readout (stable). Keep learned routing (the router gets X,
# same as V1 -- it's the messages that become state-aware, not the routing).


class HierarchicalWTABlockV4(nn.Module):
    """Block with edge-aware source-state messages.

    Forward signature extended: takes src_ids (B, L) = source node IDs for
    each edge token (plus query token's source, which is qsrc). Messages are
    msg_fn(cat[X, S[src_ids]]), carrying both edge content and source slot
    state. Routing stays static (router sees X).
    """

    def __init__(self, n_groups, group_size, d_slot):
        super().__init__()
        self.n_groups = n_groups
        self.group_size = group_size
        self.n_slots = n_groups * group_size
        self.d_slot = d_slot

        # Routing: same as V1 (X only)
        self.group_proj = nn.Linear(d_slot, n_groups, bias=False)
        self.slot_proj = nn.Linear(d_slot, group_size, bias=False)

        # Messages: now take [X, S[src]] -> 2d input
        self.msg_fn = nn.Sequential(
            nn.Linear(2 * d_slot, d_slot * 2), nn.GELU(),
            nn.Linear(d_slot * 2, d_slot))
        self.gate = nn.Sequential(
            nn.Linear(2 * d_slot, d_slot), nn.GELU(),
            nn.Linear(d_slot, 1))

        # Update unchanged
        self.update = nn.Sequential(
            nn.Linear(d_slot * 2, d_slot * 2), nn.GELU(),
            nn.Linear(d_slot * 2, d_slot))
        self.ln = nn.LayerNorm(d_slot)

        # Residual init: zero the source-state half of msg/gate input
        # projections so V4 STARTS as V1 (empty-state messages = X-only
        # messages) and grows source-state dependence via gradient descent.
        with torch.no_grad():
            self.msg_fn[0].weight[:, d_slot:].zero_()
            self.gate[0].weight[:, d_slot:].zero_()

    def forward(self, X, S, src_ids, tau=0.1):
        B, L, d = X.shape
        N = self.n_slots
        G = self.n_groups
        K = self.group_size

        # EDGE-AWARE: gather source slot states
        src_ids_clamped = src_ids.clamp(0, N - 1)
        src_expand = src_ids_clamped.unsqueeze(-1).expand(-1, -1, d)   # (B, L, d)
        src_states = torch.gather(S, 1, src_expand)                     # (B, L, d)

        # Messages now depend on edge content AND source slot state
        XS = torch.cat([X, src_states], dim=-1)                         # (B, L, 2d)
        msg = self.msg_fn(XS)                                            # (B, L, d)
        gate = torch.sigmoid(self.gate(XS))                              # (B, L, 1)
        gated_msg = msg * gate

        # Routing: same as V1 (sees X only)
        group_scores = self.group_proj(X)
        slot_scores = self.slot_proj(X)

        if self.training:
            group_assign = F.gumbel_softmax(group_scores, tau=tau, hard=True, dim=-1)
            slot_assign = F.gumbel_softmax(slot_scores, tau=tau, hard=True, dim=-1)
        else:
            group_idx = group_scores.argmax(dim=-1)
            slot_idx = slot_scores.argmax(dim=-1)
            group_assign = F.one_hot(group_idx, G).float()
            slot_assign = F.one_hot(slot_idx, K).float()

        full_assign = group_assign.unsqueeze(-1) * slot_assign.unsqueeze(-2)
        full_assign = full_assign.view(B, L, N)

        incoming = torch.bmm(full_assign.transpose(1, 2), gated_msg)

        S = S + self.update(torch.cat([S, incoming], dim=-1))
        S = self.ln(S)
        return S


class HierarchicalCircuitV4(nn.Module):
    """Graph-traversal circuit with edge-aware source-state messages."""

    def __init__(self, n_groups=16, group_size=16, d_slot=256,
                 n_steps=24, n_relations=2, max_nodes=50):
        super().__init__()
        self.n_groups = n_groups
        self.group_size = group_size
        self.n_slots = n_groups * group_size
        self.d_slot = d_slot
        self.n_steps = n_steps

        self.node_embed = nn.Embedding(max_nodes, d_slot)
        self.rel_embed = nn.Embedding(n_relations, d_slot)
        self.edge_enc = nn.Sequential(
            nn.Linear(3 * d_slot, d_slot), nn.GELU(), nn.Linear(d_slot, d_slot))
        self.query_enc = nn.Sequential(
            nn.Linear(2 * d_slot, d_slot), nn.GELU(), nn.Linear(d_slot, d_slot))

        self.block = HierarchicalWTABlockV4(n_groups, group_size, d_slot)

        self.S_init = nn.Parameter(torch.randn(n_groups * group_size, d_slot) * 0.02)

        self.head = nn.Sequential(
            nn.Linear(d_slot * 2, d_slot), nn.GELU(), nn.Linear(d_slot, 2))

        self.register_buffer('tau', torch.tensor(0.5))

    def forward(self, src, rel, tgt, mask, qsrc, qtgt):
        B = src.shape[0]
        L_edges = src.shape[1]
        N = self.n_slots
        d = self.d_slot

        src_emb = self.node_embed(src.clamp(0, 49))
        tgt_emb = self.node_embed(tgt.clamp(0, 49))
        rel_emb = self.rel_embed(rel)
        X = self.edge_enc(torch.cat([src_emb, rel_emb, tgt_emb], dim=-1))   # (B, L, d)

        q_tok = self.query_enc(torch.cat([
            self.node_embed(qsrc.clamp(0, 49)),
            self.node_embed(qtgt.clamp(0, 49))], dim=-1))
        X = torch.cat([X, q_tok.unsqueeze(1)], dim=1)                       # (B, L+1, d)

        # Source IDs for every token in X: edge sources + query source
        src_ids = torch.cat([src, qsrc.unsqueeze(1)], dim=1)                # (B, L+1)

        S = self.S_init.unsqueeze(0).expand(B, -1, -1).contiguous()

        tau_val = self.tau.item()
        for step in range(self.n_steps):
            S = self.block(X, S, src_ids=src_ids, tau=tau_val)

        # V1 gather readout (stable)
        s_src = torch.gather(S, 1, qsrc.clamp(0, N-1).unsqueeze(1).unsqueeze(2).expand(-1, -1, d)).squeeze(1)
        s_tgt = torch.gather(S, 1, qtgt.clamp(0, N-1).unsqueeze(1).unsqueeze(2).expand(-1, -1, d)).squeeze(1)
        logits = self.head(torch.cat([s_src, s_tgt], dim=-1))
        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_eval(model, arch_name, n_steps, batch_size, device,
                   train_depths, test_depths, anneal_tau=False):
    n_params = count_params(model)
    base_lr = 5e-5 if n_params > 500000 else 1e-4
    warmup = 1000

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    model.train()

    for step in range(1, n_steps + 1):
        if step < warmup:
            lr = base_lr * step / warmup
        else:
            progress = (step - warmup) / max(n_steps - warmup, 1)
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Anneal tau: start soft, end hard
        if anneal_tau and hasattr(model, 'tau'):
            tau = max(0.1, 1.0 - 0.9 * step / n_steps)
            model.tau.fill_(tau)

        src, rel, tgt, mask, qsrc, qtgt, labels = make_graph_batch(
            batch_size, train_depths, device=device)
        logits = model(src, rel, tgt, mask, qsrc, qtgt)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0:
            acc = (logits.argmax(-1) == labels).float().mean().item()
            tau_val = model.tau.item() if hasattr(model, 'tau') else 0
            print(f"  [{arch_name}] step {step}/{n_steps} loss={loss.item():.4f} "
                  f"acc={acc:.3f} lr={lr:.6f} tau={tau_val:.3f}", flush=True)

    # Thorough eval
    model.eval()
    if hasattr(model, 'tau'):
        model.tau.fill_(0.1)  # hard routing at eval

    results = {}
    with torch.no_grad():
        for label, depths in [('train', train_depths), ('ood', test_depths)]:
            total_correct = 0
            total = 0
            for d in depths:
                for _ in range(5):
                    src, rel, tgt, mask, qsrc, qtgt, labels = make_graph_batch(
                        200, [d], device=device)
                    logits = model(src, rel, tgt, mask, qsrc, qtgt)
                    total_correct += (logits.argmax(-1) == labels).sum().item()
                    total += 200
            results[f'{label}_acc'] = total_correct / total

        for d in test_depths:
            correct = 0
            total = 0
            for _ in range(5):
                src, rel, tgt, mask, qsrc, qtgt, labels = make_graph_batch(
                    200, [d], device=device)
                logits = model(src, rel, tgt, mask, qsrc, qtgt)
                correct += (logits.argmax(-1) == labels).sum().item()
                total += 200
            results[f'depth_{d}'] = correct / total

    return results


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def main(smoke=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path('results/hierarchical_wta')
    out_dir.mkdir(parents=True, exist_ok=True)

    train_depths = [3, 4, 5, 6, 7, 8]
    test_depths = [3, 5, 8, 12, 20, 30]
    n_steps = 10000 if not smoke else 1000
    batch_size = 128 if not smoke else 32

    print("=" * 60)
    print("HIERARCHICAL WTA: Matched Params Head-to-Head")
    print("  16 groups x 16 slots = 256 total, 16-way competition max")
    print(f"  Steps: {n_steps}, Batch: {batch_size}")
    print("=" * 60)

    all_results = {}

    # Hierarchical Circuit: 16 groups x 16 slots, d=256, 32 steps
    circuit = HierarchicalCircuit(
        n_groups=16, group_size=16, d_slot=256, n_steps=32).to(device)
    c_params = count_params(circuit)
    print(f"\nHierarchical Circuit: {c_params:,} params")
    print(f"  16 groups x 16 slots = 256 slots")
    print(f"  Max competition: 16-way (both levels)")

    t0 = time.time()
    c_results = train_and_eval(circuit, 'h-circuit', n_steps, batch_size, device,
                                train_depths, test_depths, anneal_tau=True)
    c_time = time.time() - t0
    c_results['params'] = c_params
    c_results['time'] = c_time
    all_results['h_circuit'] = c_results

    print(f"\n  H-Circuit: train={c_results['train_acc']:.3f} ood={c_results['ood_acc']:.3f} "
          f"time={c_time:.0f}s")
    for d in test_depths:
        print(f"    depth {d}: {c_results[f'depth_{d}']:.3f}")

    del circuit
    torch.cuda.empty_cache()

    # Transformer (same param budget)
    transformer = ScalableTransformer(256, 8, 12, task_type='graph').to(device)
    t_params = count_params(transformer)
    print(f"\nTransformer: {t_params:,} params (d=256, h=8, L=12)")

    t0 = time.time()
    t_results = train_and_eval(transformer, 'transformer', n_steps, batch_size, device,
                                train_depths, test_depths)
    t_time = time.time() - t0
    t_results['params'] = t_params
    t_results['time'] = t_time
    all_results['transformer'] = t_results

    print(f"\n  Transformer: train={t_results['train_acc']:.3f} ood={t_results['ood_acc']:.3f} "
          f"time={t_time:.0f}s")
    for d in test_depths:
        print(f"    depth {d}: {t_results[f'depth_{d}']:.3f}")

    del transformer
    torch.cuda.empty_cache()

    # Head-to-head
    print(f"\n{'='*60}")
    print("HEAD-TO-HEAD: HIERARCHICAL WTA vs TRANSFORMER")
    print(f"{'='*60}")
    print(f"{'':>10s}  {'H-Circuit':>10s}  {'Transformer':>12s}  {'Delta':>8s}")
    print(f"{'Params':>10s}  {c_params:>10,}  {t_params:>12,}")
    print(f"{'Train':>10s}  {c_results['train_acc']:>10.3f}  {t_results['train_acc']:>12.3f}")

    c_ood = c_results['ood_acc']
    t_ood = t_results['ood_acc']
    delta = c_ood - t_ood
    winner = "H-CIRCUIT" if delta > 0.02 else ("TIE" if abs(delta) < 0.02 else "TRANSFORMER")
    print(f"{'OOD':>10s}  {c_ood:>10.3f}  {t_ood:>12.3f}  {delta:>+8.3f}  {winner}")

    print(f"\nPer-depth:")
    print(f"{'Depth':>6s}  {'H-Circuit':>10s}  {'Transformer':>12s}  {'Delta':>8s}  {'Winner':>10s}")
    print("-" * 52)
    c_wins = 0
    t_wins = 0
    for d in test_depths:
        c_d = c_results[f'depth_{d}']
        t_d = t_results[f'depth_{d}']
        dd = c_d - t_d
        w = "H-CIRCUIT" if dd > 0.02 else ("TIE" if abs(dd) < 0.02 else "TF")
        if dd > 0.02: c_wins += 1
        elif dd < -0.02: t_wins += 1
        marker = " *" if d > 8 else ""
        print(f"{d:>6d}  {c_d:>10.3f}  {t_d:>12.3f}  {dd:>+8.3f}  {w:>10s}{marker}")

    print(f"\n--- VERDICT ---")
    print(f"  H-Circuit wins: {c_wins}/{len(test_depths)} depths")
    print(f"  Transformer wins: {t_wins}/{len(test_depths)} depths")

    # Training stability check
    print(f"\n  Training stability:")
    print(f"    H-Circuit: {'STABLE' if c_results['train_acc'] > 0.95 else 'UNSTABLE'} "
          f"(final train acc: {c_results['train_acc']:.3f})")
    print(f"    Transformer: {'STABLE' if t_results['train_acc'] > 0.95 else 'UNSTABLE'} "
          f"(final train acc: {t_results['train_acc']:.3f})")

    if c_ood > t_ood + 0.02:
        print(f"\n  >>> HIERARCHICAL WTA WINS AT MATCHED PARAMS. CIRCUITS SCALE.")
    elif abs(c_ood - t_ood) <= 0.02:
        print(f"\n  >>> TIE AT MATCHED PARAMS. Circuit matches TF with pure discrete routing.")
    else:
        print(f"\n  >>> TRANSFORMER STILL WINS (+{t_ood-c_ood:.3f}). Need more work.")

    # Save
    def sanitize(obj):
        if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list): return [sanitize(v) for v in obj]
        if isinstance(obj, torch.Tensor): return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, float) and (obj != obj or abs(obj) == float('inf')): return str(obj)
        return obj

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(sanitize(all_results), f, indent=2)
    print(f"\nResults saved to {out_dir / 'results.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    main(smoke=args.smoke)
