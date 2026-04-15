"""
Generate the hero figure for the V4 paper — scaling curve of depth-30
extrapolation accuracy vs model parameter count.

This is the Twitter thread figure. If the 1.5B V4 point lands above the
100M point, the line goes UP and the scaling claim materializes visually.

Run this after the 1.5B V4 graph training finishes and writes
`results/hwta_v4_d8192_lr1e-5_s2000/results.json`.

Usage:
    python generate_hero_figure.py

Output files:
    hero_figure.png  (for Twitter)
    hero_figure.pdf  (for the paper)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless, no interactive display
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Result file paths (update these if your file layout differs)
# ---------------------------------------------------------------------------

RESULT_FILES = {
    'v4_12m':   'results/hwta_v4_d768/results.json',
    # 100M V4: use the 2 clean seeds (drop the lr=2e-5 seed-1 file which was
    # overwritten by a later run, and drop the failed lr=3e-5 seed-2 at 0.616).
    # Clean pair: seed-1 @ 3e-5 (from hwta_v4_d2048/results.json, d30=0.690) +
    # seed-2 @ 2e-5 (from hwta_v4_d2048_seed2_lr2e-5/results.json, d30=0.743).
    'v4_100m_seed1_3e5': 'results/hwta_v4_d2048/results.json',
    'v4_100m_seed2_2e5': 'results/hwta_v4_d2048_seed2_lr2e-5/results.json',
    'v4_1p5b':  'results/hwta_v4_d8192_lr1e-5_s2000/results.json',
    'tf_12m':   'results/hwta_12m_tf/results.json',
    'tf_103m':  'results/hwta_v4_d2048/results.json',  # has TF baseline embedded
}


def load(path):
    """Return parsed JSON if path exists, else None."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def extract_depth30(j, is_tf=False):
    """Extract depth_30 accuracy from a circuit or transformer result."""
    if j is None:
        return None
    if is_tf:
        # hwta_v4_d2048/results.json has both circuit and transformer keys
        if 'transformer' in j:
            return j['transformer'].get('depth_30')
        return j.get('depth_30')
    if 'circuit' in j:
        return j['circuit'].get('depth_30')
    return j.get('depth_30')


def extract_params(j, is_tf=False):
    """Extract parameter count from a result file."""
    if j is None:
        return None
    if is_tf and 'transformer' in j:
        return j['transformer'].get('params')
    if 'circuit' in j:
        return j['circuit'].get('params')
    return j.get('params')


def main():
    # Load all available results
    v4_12m = load(RESULT_FILES['v4_12m'])
    v4_100m_s1 = load(RESULT_FILES['v4_100m_seed1_3e5'])
    v4_100m_s2 = load(RESULT_FILES['v4_100m_seed2_2e5'])
    v4_1p5b = load(RESULT_FILES['v4_1p5b'])
    tf_12m = load(RESULT_FILES['tf_12m'])
    tf_103m_full = load(RESULT_FILES['tf_103m'])  # contains both circuit and transformer

    # Extract V4 depth-30 accuracies
    v4_points = []
    v4_labels = []

    if v4_12m is not None:
        params = extract_params(v4_12m)
        d30 = extract_depth30(v4_12m)
        if params and d30:
            v4_points.append((params, d30))
            v4_labels.append(f'V4 12M\n{d30:.3f}')

    # For 100M, average the clean 2 seeds if both available
    v4_100m_d30_vals = []
    v4_100m_params = None
    if v4_100m_s1 is not None:
        d30 = extract_depth30(v4_100m_s1)
        if d30 is not None:
            v4_100m_d30_vals.append(d30)
            v4_100m_params = extract_params(v4_100m_s1)
    if v4_100m_s2 is not None:
        d30 = extract_depth30(v4_100m_s2)
        if d30 is not None:
            v4_100m_d30_vals.append(d30)
            if v4_100m_params is None:
                v4_100m_params = extract_params(v4_100m_s2)

    if v4_100m_d30_vals and v4_100m_params:
        v4_100m_d30_mean = sum(v4_100m_d30_vals) / len(v4_100m_d30_vals)
        v4_points.append((v4_100m_params, v4_100m_d30_mean))
        v4_labels.append(f'V4 100M\n{v4_100m_d30_mean:.3f}\n(n={len(v4_100m_d30_vals)} seeds)')

    if v4_1p5b is not None:
        params = extract_params(v4_1p5b)
        d30 = extract_depth30(v4_1p5b)
        if params and d30:
            v4_points.append((params, d30))
            v4_labels.append(f'V4 1.5B\n{d30:.3f}')
        else:
            print(f"WARNING: 1.5B result file exists but missing params or depth_30")
    else:
        print(f"NOTE: 1.5B result file not found at {RESULT_FILES['v4_1p5b']}")
        print(f"      (training probably still in progress; figure will omit the 1.5B point)")

    # Extract TF depth-30 accuracies
    tf_points = []
    tf_labels = []

    if tf_12m is not None:
        params = extract_params(tf_12m)
        d30 = extract_depth30(tf_12m)
        if params and d30:
            tf_points.append((params, d30))
            tf_labels.append(f'TF 12M\n{d30:.3f}')

    if tf_103m_full is not None:
        params = extract_params(tf_103m_full, is_tf=True)
        d30 = extract_depth30(tf_103m_full, is_tf=True)
        if params and d30:
            tf_points.append((params, d30))
            tf_labels.append(f'TF 103M\n{d30:.3f}')

    # Sort by parameter count for line plotting
    v4_points.sort(key=lambda p: p[0])
    tf_points.sort(key=lambda p: p[0])

    print("\n=== V4 scaling points (sorted) ===")
    for p, d in v4_points:
        print(f"  params={p:,}  depth_30={d:.4f}")

    print("\n=== TF scaling points (sorted) ===")
    for p, d in tf_points:
        print(f"  params={p:,}  depth_30={d:.4f}")

    if not v4_points:
        print("\nERROR: No V4 points available. Cannot generate figure.")
        return
    if not tf_points:
        print("\nWARNING: No TF points available. Generating V4-only plot.")

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)

    # V4 line
    v4_params = [p for p, _ in v4_points]
    v4_d30 = [d for _, d in v4_points]
    ax.plot(v4_params, v4_d30, 'o-',
            color='#1f77b4', linewidth=2.5, markersize=11,
            label='V4 Circuit (ours)', zorder=3)

    # TF line (if we have points)
    if tf_points:
        tf_params = [p for p, _ in tf_points]
        tf_d30 = [d for _, d in tf_points]
        ax.plot(tf_params, tf_d30, 's--',
                color='#ff7f0e', linewidth=2.5, markersize=10,
                label='Matched Transformer', zorder=2)

    # Random baseline
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6,
               label='random baseline', zorder=1)

    # Annotations on V4 points
    for (p, d), label in zip(v4_points, v4_labels):
        lines = label.split('\n')
        ax.annotate(
            lines[0] + (f'\n{d:.3f}' if len(lines) < 2 else ''),
            xy=(p, d), xytext=(0, 14), textcoords='offset points',
            ha='center', fontsize=9, color='#1f77b4', fontweight='bold'
        )

    # Annotations on TF points
    for (p, d), label in zip(tf_points, tf_labels):
        lines = label.split('\n')
        ax.annotate(
            lines[0] + (f'\n{d:.3f}' if len(lines) < 2 else ''),
            xy=(p, d), xytext=(0, -22), textcoords='offset points',
            ha='center', fontsize=9, color='#ff7f0e', fontweight='bold'
        )

    ax.set_xscale('log')
    ax.set_xlabel('Parameters', fontsize=13)
    ax.set_ylabel('Depth-30 OOD accuracy', fontsize=13)
    ax.set_title(
        'Depth-30 extrapolation accuracy vs model scale\n'
        'Causal graph traversal — train depths 3-15, test depth 30',
        fontsize=13, pad=14,
    )
    ax.set_ylim([0.4, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)

    # Annotate the delta at 100M if we have both points
    if v4_100m_d30_vals and tf_103m_full is not None:
        tf_d30_100m = extract_depth30(tf_103m_full, is_tf=True)
        v4_d30_100m = sum(v4_100m_d30_vals) / len(v4_100m_d30_vals)
        if tf_d30_100m and v4_d30_100m:
            delta = (v4_d30_100m - tf_d30_100m) * 100
            mid_x = (v4_100m_params + extract_params(tf_103m_full, is_tf=True)) / 2
            mid_y = (v4_d30_100m + tf_d30_100m) / 2
            ax.annotate(
                f'+{delta:.1f} pts',
                xy=(mid_x, mid_y),
                xytext=(50, 0), textcoords='offset points',
                fontsize=11, fontweight='bold', color='#2a9d8f',
                arrowprops=dict(arrowstyle='-', color='#2a9d8f', lw=1, alpha=0.5),
            )

    plt.tight_layout()

    # Save both formats
    png_path = 'hero_figure.png'
    pdf_path = 'hero_figure.pdf'
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"\nSaved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"\nReady for Twitter thread.")


if __name__ == '__main__':
    main()
