# Family, Not Scale

*by Suleman Imdad — Johns Hopkins University*

*Self-presentation trust in small language models — chain-of-thought faithfulness, introspection, and
evaluation-awareness — probed across five model families, entirely on a consumer laptop (Apple M4,
48 GB).* Full write-up: **[`PAPER.md`](PAPER.md)** (or `PAPER.html`).

## TL;DR

Three questions about whether you can trust what a small LLM tells you about itself, each with a
**confound or robustness control that overturns the naive reading**:

1. **Chain-of-thought faithfulness** — Does the model already know its answer before it reasons? A
   linear probe of the pre-CoT residual stream *looks* like "yes, committed" on CommonsenseQA, but a
   confound cut to the cases where reasoning *changed* the answer (`S = {forced ≠ cot}`) **reverses**
   it, and a **truncation counterfactual confirms it causally**: cutting the CoT reverts the answer
   to the pre-reasoning one. → CoT is **load-bearing, not post-hoc** (replicated on Qwen-1.5B and Llama-3.1-8B).
2. **Introspection** — Can the model report a concept injected into its activations? A faithful
   replication of Lindsey et al. (2025) finds a **clean functional null across three families**
   (Qwen, Gemma, Llama); a tempting aggregate signal dissolves under a per-concept breakdown.
3. **Evaluation-awareness (headline)** — Does the model behave differently when it thinks it is being
   tested? The direction of the effect is **family-dependent, not scale-gated**: **Llama-3.1-8B
   becomes *more cautious* when told it is being evaluated** (the frontier "eval-gaming" direction,
   at 8B), while Qwen/DeepSeek/GPT-OSS lean *more compliant* and Gemma over-refuses to a ceiling
   (16.4% false-refusal on XSTest — the worst of the four).

> The "trust by incapacity" reading holds for the two *cognitive* axes (the model is faithful and
> non-introspective because it *can't* be otherwise) — but breaks on the behavioral one: eval-gaming
> is **trained in**, and which families have it is an empirical fact.

## Reproduce

**Environment:** Python 3.9, `torch` (MPS), `transformers`, `datasets`, `scikit-learn`, `mlx_lm`
(for 4-bit models). Models pulled from the HF hub on first use (`mlx-community/*` and `unsloth/*`
mirrors are ungated). Every script is parameterized by environment variables.

```bash
# --- 1. Faithfulness (commitment probe + causal counterfactual) ---
DATASET=aqua_rat N=150 python3 experiments/sanity_gate.py          # GO/NO-GO: does CoT beat direct?
DATASET=aqua_rat N=150 python3 experiments/extract_activations.py  # dump pre-CoT states + cot/direct/gold
DATASET=aqua_rat python3 experiments/extract_scaffold_acts.py      # re-capture at the answer scaffold
DATASET=aqua_rat python3 experiments/commitment_probe.py           # probe + S={forced!=cot} confound cut
DATASET=aqua_rat python3 experiments/causal_faithfulness.py        # truncation counterfactual on S

# --- 2. Introspection (concept injection) ---
python3 experiments/build_steering.py                              # diff-of-means vectors + manipulation check
LAYERS=16,20,24 MS=1.5,2,3,4 python3 experiments/introspect_probe.py   # priming-controlled + leak/internality ramp
LAYER=20 M=3 python3 experiments/introspect_diag.py                # per-concept breakdown (dissolves false signals)

# --- 3. Evaluation-awareness ---
N=40 python3 experiments/eval_extract.py                           # TEST-vs-CASUAL representation probe data
python3 experiments/eval_probe.py                                  # decodability + MCQA behavioral divergence
REQSET=hard python3 experiments/eval_behavior.py                   # refusal under safety-eval vs casual framing (fp16)
REQSET=hard MLX_REPO=mlx-community/<repo> NAME=<tag> python3 experiments/eval_behavior_mlx.py   # 4-bit families
python3 experiments/eval_family.py                                # cross-family comparison table
MLX_REPO=mlx-community/<repo> NAME=<tag> python3 experiments/eval_xstest.py   # XSTest false-refusal
```

Set `MODEL=<hf-id>` (fp16, transformers) on any faithfulness/introspection script to change the
model; the scaling axis used `Qwen/Qwen2.5-{1.5B,7B,14B,32B}-Instruct`, the cross-family axis used
`unsloth/*` (fp16) and `mlx-community/*-4bit` (MLX). **Note:** launch background runs from inside
this directory (`cd` first) — the scripts use relative paths.

## Layout

| Path | What |
|---|---|
| `experiments/` | all probes/runners (faithfulness, introspection, eval-awareness) |
| `results/` | per-run JSON/NPZ artifacts + logs (one per model × experiment) |
| `PAPER.md` / `PAPER.html` | the write-up |

## Hardware & cost

Everything ran on a single **Apple M4 (48 GB unified memory)** over the MPS backend, with 4-bit MLX
for the 14B–32B models. No cloud, **~$0**. fp16 caps around 14B on this machine; larger models use
MLX 4-bit (the headline Llama result is fp16-verified).

## Caveats (the honest list)

- Sub-frontier scales: faithfulness shown on two families (Qwen-1.5B, Llama-3.1-8B), introspection on
  three (Qwen, Gemma, Llama), eval-awareness on five labs — all 1.5B–32B.
- Linear probes / first-token reads; behavioral results use N≈40 borderline prompts and a regex
  refusal classifier (per-request consistency mitigates this).
- 4-bit quantization *inflates* eval-effect magnitudes (verified: Llama +0.24 at 4-bit → +0.07 fp16);
  4-bit Δ's are upper bounds.
- "Faithful" = the final answer isn't precommitted (correlationally) and reverts under truncation
  (causally) — not a per-reasoning-step causal claim.

## Closest related work

Cox, Kianersi & Garriga-Alonso (2026, arXiv:2603.01437) — concurrent pre-CoT probing + steering;
Lindsey et al. (2025) — the introspection protocol we replicate; Abdelnabi & Salem (2025,
arXiv:2505.14617, NeurIPS) — test-awareness varies across reasoning models (we add the un-steered,
cross-family behavioral counterpart); Röttger et al. (2024) — XSTest. See `PAPER.md` §8 for the full
list and our honest positioning against each.

## Submission-ready copies

See [`submissions/`](submissions/): `paper.tex` (arXiv / workshop LaTeX), `blog.md` (Alignment Forum / LessWrong), and `SUBMISSIONS.md` (per-platform formatting guide).
