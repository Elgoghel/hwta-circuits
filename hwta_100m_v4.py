"""
100M Hierarchical WTA V4 -- With Slot Self-Attention + Attention Readout
===========================================================================
The scaling fix. V4 changes vs V2:
  1. HierarchicalCircuitV2: slots self-attend (real multi-hop) + attention
     readout (all slots contribute, not just slots 0-49).
  2. bf16 autocast (no GradScaler -- bf16 doesn't need it).
  3. torch.compile(mode='reduce-overhead') on the model.
  4. Gradient accumulation x4 (effective batch = batch_arg * 4).
  5. Fused AdamW.
  6. Train depths expanded from [3..8] to [3..15] -- forces generalization.

Entry points:
  Sanity check (12M):  python hwta_100m_v4.py --d-slot 768 --steps 800
  Full 100M run:       python hwta_100m_v4.py --d-slot 2048 --steps 2500
"""

import argparse
import math
import random
import time
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int):
    """Set all RNG seeds so the run is reproducible-ish.
    Note: make_graph_batch uses Python's random module, so that's the
    critical one. torch and numpy covered for completeness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from hierarchical_wta import (
    HierarchicalCircuitV2, HierarchicalCircuitV3, HierarchicalCircuitV4,
    count_params,
)
from scaling_laws import ScalableTransformer, make_graph_batch


def train_eval(model, arch, n_steps, micro_batch, accum_steps, device,
               train_depths, test_depths, base_lr, anneal=True, use_bf16=True):
    warmup = min(1000, n_steps // 4)
    opt_cls = torch.optim.AdamW
    try:
        optimizer = opt_cls(model.parameters(), lr=base_lr, weight_decay=0.01, fused=True)
    except (RuntimeError, TypeError):
        optimizer = opt_cls(model.parameters(), lr=base_lr, weight_decay=0.01)

    model.train()
    amp_ctx = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if use_bf16 else \
              (lambda: torch.amp.autocast('cuda', enabled=False))

    effective_batch = micro_batch * accum_steps
    log_every = max(25, n_steps // 120)

    for step in range(1, n_steps + 1):
        if step < warmup:
            lr = base_lr * step / warmup
        else:
            progress = (step - warmup) / max(n_steps - warmup, 1)
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if anneal and hasattr(model, 'tau'):
            tau = max(0.1, 1.0 - 0.9 * step / n_steps)
            model.tau.fill_(tau)

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for micro_step in range(accum_steps):
            src, rel, tgt, mask, qsrc, qtgt, labels = make_graph_batch(
                micro_batch, train_depths, device=device)
            with amp_ctx():
                logits = model(src, rel, tgt, mask, qsrc, qtgt)
                loss = F.cross_entropy(logits, labels) / accum_steps
            loss.backward()
            total_loss += loss.item() * accum_steps
            total_correct += (logits.argmax(-1) == labels).sum().item()
            total_samples += micro_batch

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0 or step == 1:
            acc = total_correct / total_samples
            tau_val = model.tau.item() if hasattr(model, 'tau') else 0
            print(f"  [{arch}] step {step}/{n_steps} loss={total_loss/accum_steps:.4f} "
                  f"acc={acc:.3f} tau={tau_val:.3f} lr={lr:.2e} effbatch={effective_batch}",
                  flush=True)

    model.eval()
    if hasattr(model, 'tau'):
        model.tau.fill_(0.1)
    results = {}
    eval_batch = 200
    with torch.no_grad():
        with (torch.autocast('cuda', dtype=torch.bfloat16) if use_bf16
              else torch.amp.autocast('cuda', enabled=False)):
            for label, depths in [('train', train_depths), ('ood', test_depths)]:
                correct = total = 0
                for d in depths:
                    for _ in range(5):
                        src, rel, tgt, mask, qsrc, qtgt, labels = make_graph_batch(
                            eval_batch, [d], device=device)
                        logits = model(src, rel, tgt, mask, qsrc, qtgt)
                        correct += (logits.argmax(-1) == labels).sum().item()
                        total += eval_batch
                results[f'{label}_acc'] = correct / total

            for d in test_depths:
                correct = total = 0
                for _ in range(5):
                    src, rel, tgt, mask, qsrc, qtgt, labels = make_graph_batch(
                        eval_batch, [d], device=device)
                    logits = model(src, rel, tgt, mask, qsrc, qtgt)
                    correct += (logits.argmax(-1) == labels).sum().item()
                    total += eval_batch
                results[f'depth_{d}'] = correct / total

    return results


def main(d_slot=2048, n_groups=16, group_size=16, n_steps_prop=24,
         n_steps=2500, micro_batch=12, accum_steps=4, n_heads=8,
         run_transformer=True, sanity_check=False, no_compile=False,
         train_depth_max=15, enable_slot_attn=True, version='v3',
         seed=42, lr_override=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(seed)
    # Build unique output tag including seed, LR, grid, and steps to avoid collisions
    out_tag = f'hwta_v4_d{d_slot}'
    # Grid suffix only when non-default 16x16
    if n_groups != 16 or group_size != 16:
        out_tag += f'_g{n_groups}x{group_size}'
    if seed != 42:
        out_tag += f'_seed{seed}'
    if lr_override is not None:
        # Preserve 1 decimal so 1.5e-5 and 2e-5 don't collide (was :.0e -> both rounded to "2e-05")
        lr_str = f'{lr_override:.1e}'          # "1.5e-05" / "2.0e-05"
        lr_str = lr_str.replace('e-0', 'e-')    # "1.5e-5" / "2.0e-5"
        lr_str = lr_str.replace('.0e', 'e')     # "1.5e-5" / "2e-5"
        lr_str = lr_str.replace('.', 'p')       # "1p5e-5" / "2e-5"
        out_tag += f'_lr{lr_str}'
    # Steps suffix only when non-default 2500 (prevents short sanity runs from
    # overwriting full training runs that share the same model config).
    if n_steps != 2500:
        out_tag += f'_s{n_steps}'
    out_dir = Path(f'results/{out_tag}')
    out_dir.mkdir(parents=True, exist_ok=True)

    train_depths = list(range(3, train_depth_max + 1))
    test_depths = [3, 5, 8, 12, 20, 30]

    print("=" * 60)
    tag = "SANITY (12M)" if sanity_check else "FULL 100M"
    print(f"V4 HIERARCHICAL WTA -- {tag}")
    print(f"  d_slot={d_slot}, {n_groups}x{group_size} slots, {n_heads} attn heads")
    print(f"  Prop={n_steps_prop}, Steps={n_steps}, MicroBatch={micro_batch}, "
          f"Accum={accum_steps}, EffBatch={micro_batch*accum_steps}")
    print(f"  Train depths: {train_depths[0]}-{train_depths[-1]} | "
          f"Test depths: {test_depths}")
    print(f"  FEATURES: slot self-attention + attention readout, bf16, "
          f"compile={'off' if no_compile else 'on'}, fused AdamW")
    print("=" * 60, flush=True)

    all_results = {}

    # ------------------------------------------------------------------
    # Circuit
    # ------------------------------------------------------------------
    if version == 'v4':
        circuit = HierarchicalCircuitV4(
            n_groups=n_groups, group_size=group_size,
            d_slot=d_slot, n_steps=n_steps_prop,
        ).to(device)
        ver_label = 'V4 (edge-aware messages)'
    elif version == 'v3':
        circuit = HierarchicalCircuitV3(
            n_groups=n_groups, group_size=group_size,
            d_slot=d_slot, n_steps=n_steps_prop,
        ).to(device)
        ver_label = 'V3 (dynamic routing)'
    else:
        circuit = HierarchicalCircuitV2(
            n_groups=n_groups, group_size=group_size,
            d_slot=d_slot, n_steps=n_steps_prop, n_heads=n_heads,
            enable_slot_attn=enable_slot_attn,
        ).to(device)
        ver_label = 'V2 (slot attention)'
    cp = count_params(circuit)
    print(f"\nH-Circuit {ver_label}: {cp:,} params", flush=True)

    # Scale LR with params (bigger model = smaller LR)
    if lr_override is not None:
        base_lr = lr_override
        print(f"  LR override: {base_lr:.2e}", flush=True)
    else:
        base_lr = 5e-5 if cp < 30_000_000 else 3e-5

    if not no_compile:
        try:
            circuit = torch.compile(circuit, mode='reduce-overhead', dynamic=False)
            print("  torch.compile: enabled", flush=True)
        except Exception as e:
            print(f"  torch.compile FAILED, falling back: {e}", flush=True)

    t0 = time.time()
    try:
        cr = train_eval(circuit, 'h-circuit', n_steps, micro_batch, accum_steps,
                        device, train_depths, test_depths, base_lr=base_lr,
                        anneal=True, use_bf16=True)
    except torch.cuda.OutOfMemoryError as e:
        print(f"\nOOM: {e}")
        print("Retry with smaller micro_batch")
        return

    cr['params'] = cp
    cr['time'] = time.time() - t0
    cr['config'] = {
        'd_slot': d_slot, 'n_groups': n_groups, 'group_size': group_size,
        'n_steps_prop': n_steps_prop, 'n_heads': n_heads,
        'micro_batch': micro_batch, 'accum_steps': accum_steps,
        'effective_batch': micro_batch * accum_steps,
        'train_depths': train_depths, 'base_lr': base_lr,
    }
    all_results['circuit'] = cr

    print(f"\n  Circuit V2: train={cr['train_acc']:.3f} ood={cr['ood_acc']:.3f}")
    for d in test_depths:
        print(f"    depth {d:>2d}: {cr[f'depth_{d}']:.3f}")
    print(f"  Time: {cr['time']:.0f}s", flush=True)
    del circuit
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Matched transformer baseline
    # ------------------------------------------------------------------
    if run_transformer:
        # Pick TF config matched to circuit params
        if cp < 30_000_000:
            tf_d, tf_h, tf_L = 384, 8, 6        # ~12M
        elif cp < 60_000_000:
            tf_d, tf_h, tf_L = 640, 8, 10       # ~53M
        else:
            tf_d, tf_h, tf_L = 896, 14, 10      # ~103M

        tf = ScalableTransformer(tf_d, tf_h, tf_L, task_type='graph').to(device)
        tp = count_params(tf)
        print(f"\nTransformer: {tp:,} params "
              f"(d={tf_d}, h={tf_h}, L={tf_L})", flush=True)

        if not no_compile:
            try:
                tf = torch.compile(tf, mode='reduce-overhead', dynamic=False)
            except Exception as e:
                print(f"  TF compile failed: {e}", flush=True)

        t0 = time.time()
        try:
            tr = train_eval(tf, 'transformer', n_steps, micro_batch, accum_steps,
                            device, train_depths, test_depths, base_lr=base_lr,
                            anneal=False, use_bf16=True)
        except torch.cuda.OutOfMemoryError:
            print("TF OOM -- skipping")
            tr = None

        if tr is not None:
            tr['params'] = tp
            tr['time'] = time.time() - t0
            tr['config'] = {'d_model': tf_d, 'n_heads': tf_h, 'n_layers': tf_L}
            all_results['transformer'] = tr
            print(f"\n  Transformer: train={tr['train_acc']:.3f} ood={tr['ood_acc']:.3f}")
            for d in test_depths:
                print(f"    depth {d}: {tr[f'depth_{d}']:.3f}")
            print(f"  Time: {tr['time']:.0f}s", flush=True)
        del tf
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"V4 HEAD-TO-HEAD")
    print(f"{'='*60}")
    print(f"  Circuit V2 ({cp:,}):     train={cr['train_acc']:.3f} ood={cr['ood_acc']:.3f}")
    if run_transformer and 'transformer' in all_results:
        tr = all_results['transformer']
        tp = tr['params']
        print(f"  Transformer ({tp:,}): train={tr['train_acc']:.3f} ood={tr['ood_acc']:.3f}")
        delta = cr['ood_acc'] - tr['ood_acc']
        winner = "CIRCUIT" if delta > 0.02 else ("TIE" if abs(delta) < 0.02 else "TRANSFORMER")
        print(f"  Delta: {delta:+.3f}  [{winner}]", flush=True)
        print("\n  Per-depth:")
        print(f"  {'depth':>6s}  {'Circuit':>9s}  {'TF':>9s}  {'delta':>9s}")
        for d in test_depths:
            c = cr.get(f'depth_{d}', 0)
            t = tr.get(f'depth_{d}', 0)
            print(f"  {d:>6d}  {c:>9.3f}  {t:>9.3f}  {c-t:>+9.3f}")

    # Compare against prior V1 results if they exist
    v1_12m_path = Path('results/hwta_10m/results.json')
    v1_12m_tf_path = Path('results/hwta_12m_tf/results.json')
    if v1_12m_path.exists():
        with open(v1_12m_path) as f:
            v1 = json.load(f)
        print(f"\n  V1 reference (12M circuit): train={v1['train_acc']:.3f} "
              f"ood={v1['ood_acc']:.3f} depth_30={v1.get('depth_30', 0):.3f}")

    def sanitize(obj):
        if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list): return [sanitize(v) for v in obj]
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        return obj

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(sanitize(all_results), f, indent=2)
    print(f"\nResults saved to {out_dir / 'results.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d-slot', type=int, default=2048)
    parser.add_argument('--n-groups', type=int, default=16)
    parser.add_argument('--group-size', type=int, default=16)
    parser.add_argument('--prop-steps', type=int, default=24)
    parser.add_argument('--steps', type=int, default=2500)
    parser.add_argument('--micro-batch', type=int, default=12)
    parser.add_argument('--accum', type=int, default=4)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--no-tf', action='store_true')
    parser.add_argument('--sanity', action='store_true',
                        help='12M sanity check config')
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--train-depth-max', type=int, default=15)
    parser.add_argument('--no-slot-attn', action='store_true',
                        help='V2 only: disable slot self-attention')
    parser.add_argument('--version', type=str, default='v4', choices=['v2', 'v3', 'v4'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=None, help='Override base LR')
    args = parser.parse_args()

    # Sanity check overrides: use 12M-sized config
    if args.sanity:
        args.d_slot = 768
        args.steps = min(args.steps, 1000)
        args.micro_batch = 24
        args.accum = 2

    main(
        d_slot=args.d_slot,
        n_groups=args.n_groups,
        group_size=args.group_size,
        n_steps_prop=args.prop_steps,
        n_steps=args.steps,
        micro_batch=args.micro_batch,
        accum_steps=args.accum,
        n_heads=args.heads,
        run_transformer=not args.no_tf,
        sanity_check=args.sanity,
        no_compile=args.no_compile,
        train_depth_max=args.train_depth_max,
        enable_slot_attn=not args.no_slot_attn,
        version=args.version,
        seed=args.seed,
        lr_override=args.lr,
    )
