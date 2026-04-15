# Attention Is Not All You Need

**Hierarchical Winner-Take-All (HWTA) Circuits for Compositional Reasoning**

HWTA is a new architecture built entirely on discrete routing over a fixed slot bank, with no softmax attention anywhere in the block. On five compositional reasoning benchmarks, it beats matched-parameter transformers by large margins.

**Headline result.** On the published CLUTRR `gen_train234_test2to10` multi-hop reasoning benchmark with **1.01x matched parameters and an identical training loop**, HWTA (272K params) beats a tree transformer (268K params) by **+44.0 points** at k=10 across 3 seeds.

## Paper

[PAPER.pdf](PAPER.pdf) - 15 pages, full methodology, ablations, and caveats.

## The numbers

| Benchmark | HWTA | Matched TF | Delta |
|---|---|---|---|
| **CLUTRR k=10** (272K vs 268K, 3 seeds) | **0.590 +/- 0.095** | 0.150 +/- 0.026 | **+44.0** |
| **CLUTRR overall** (272K vs 268K, 3 seeds) | **0.815 +/- 0.019** | 0.378 +/- 0.019 | +43.7 |
| SCAN add-jump (164 params vs 6.5M) | **100.0 +/- 0.0** (5 seeds) | 6.4 +/- 9.0 (2 seeds) | +93.6 |
| SCAN add-jump (~90K vs 846K) | **99.1 +/- 1.8** (4 seeds) | 20.4 +/- 16.3 (4 seeds) | +78.7 |
| CruxMini nested mul (3K vs 804K, 4 seeds) | **100.0 +/- 0.0** | 15.2 +/- 0.5 | +84.8 |
| ListOps (4K vs 804K, 3 seeds) | **100.0 +/- 0.0** | 41.7 +/- 0.5 | +58.3 |
| Graph 100M OOD (n=3) | 0.902 | TF 103M: 0.868 | +3.4 |
| Graph 100M depth-30 extrapolation (n=3) | 0.696 | TF 103M: 0.605 | +9.2 |
| Graph 1.5B OOD (compressed schedule, n=1) | 0.884 | - | - |

All results are reported as mean +/- std across multiple seeds with identical training loops where applicable. See [PAPER.pdf](PAPER.pdf) Section 4 for full methodology.

**Reproduce the table in one command:**

```bash
python aggregate_results.py
```

## The architectural insight in one paragraph

Early HWTA variants failed at depth generalization because **slots don't communicate**: messages were pure functions of input tokens, so 24 propagation steps iterated the same static operation on independent slots. The fix is **one line of code**: messages carry source-slot state via `torch.gather(S, src_ids)`. Every edge (A -> B) now delivers a message that depends on slot A's current state. Over 24 propagation steps, information flows along graph edges through the routing structure, and real multi-hop reasoning emerges. See [PAPER.pdf](PAPER.pdf) Section 5.1 for the HWTA-v1 through HWTA-v4 ablation.

The same content-structure separation principle instantiates differently per task: the graph HWTA uses edge-aware gather-scatter; SCAN uses positional slot buffers plus tied action embeddings; CruxMini and ListOps use a bilinear op-table indexed by op-type. In all cases, **the learnable component never touches content nonlinearly**. Structure is learned; content passes through.

## Repo layout

```
hierarchical_wta.py       HWTA core block (V1 through V4 variants)
hwta_100m_v4.py           Graph traversal training (12M to 1.5B)
scan_v4b.py               SCAN HWTA-sym (164-parameter symbolic variant)
scan_v4c.py               SCAN HWTA-learn (~90K fully-learned variant)
scan_tree_tf.py           Matched tree transformer baseline for SCAN
cruxmini.py               CruxMini nested-arithmetic HWTA circuit (3K params)
cruxmini_tf.py            Matched tree transformer baseline for CruxMini
listops.py                ListOps HWTA circuit (4K params)
listops_tf.py             Matched tree transformer baseline for ListOps
clutrr_v4.py              CLUTRR edge-aware HWTA (272K params)
train_clutrr_tf.py        Matched tree transformer baseline for CLUTRR (268K)
aggregate_results.py      Regenerates the table above from result JSONs
generate_hero_figure.py   Regenerates the scaling figure
results/                  Multi-seed result JSONs for every run in the paper
PAPER.pdf                 The paper (15 pages)
```

## Quickstart - reproduce the 164-parameter hero result

```bash
# SCAN HWTA-sym: ~30 seconds on any GPU, hits 100% compositional generalization
python scan_v4b.py --seed 42
python scan_v4b.py --seed 2
python scan_v4b.py --seed 7
python scan_v4b.py --seed 13
python scan_v4b.py --seed 99

# Matched tree transformer baseline for contrast
python scan_tree_tf.py --seed 42
python scan_tree_tf.py --seed 2
python scan_tree_tf.py --seed 7
python scan_tree_tf.py --seed 13

# Regenerate the comparison table
python aggregate_results.py
```

Expected: HWTA-sym hits 100% jump_comp within 250 training steps across all 5 seeds. The matched TF caps around 20% with high seed variance.

## Honest caveats

- **5 benchmarks.** 4 compositional generalization (SCAN, CruxMini, ListOps, Graph) + 1 published multi-hop reasoning (CLUTRR). No language modeling, no GSM8k, no full CRUXEval. Those are future work.
- **SCAN HWTA-sym is near-symbolic** (164 params). Only `action_embed` and a tiny twice/thrice projection are learned; modifier/combinator ops are deterministic. The fully-learned HWTA-learn variant (~90K params) hits 99.1% across 4 seeds, which confirms the architectural insight survives optimization.
- **1.5B run used a compressed training schedule.** At 1.5B parameters, HWTA maintains the OOD-average advantage and improves depth-20 extrapolation, but depth-30 extrapolation regressed below the 100M result. This is an open question requiring lab-level compute to resolve (longer training plus a larger Gumbel anneal window).
- **CLUTRR uses 3 seeds per configuration**, identical training loops for HWTA and matched TF. See [PAPER.pdf](PAPER.pdf) Section 4.5 for exact hyperparameters.

## What I can't do from my dorm room

I can't run HWTA at the scale of a frontier lab. The 1.5B graph scaling data point is the largest HWTA we've trained. Applying HWTA to real language modeling, real code execution, or real math reasoning would require compute well beyond one GPU. **If you're at a frontier lab and want to collaborate, reach out.**

## Credit

Marwan Khaled Elgoghel, undergraduate at Monmouth University. Independent research, 3 months, one GPU, one dorm room. Feedback welcome.

## Citation

```
@misc{elgoghel2026hwta,
  author = {Elgoghel, Marwan Khaled},
  title  = {Attention Is Not All You Need: Hierarchical WTA Circuits
            for Compositional Reasoning},
  year   = {2026},
  url    = {https://github.com/Elgoghel/hwta-circuits},
}
```

## License

MIT. See [LICENSE](LICENSE).
