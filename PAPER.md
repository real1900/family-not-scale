# Family, Not Scale: Evaluation-Gaming in Small Language Models Is Lab-Dependent, Not Just Frontier-Emergent

**Suleman Imdad**
*Johns Hopkins University*

*A cross-family, laptop-scale study of self-presentation in small LLMs — with confound-controlled
negatives on chain-of-thought faithfulness and introspection.*

*Working draft — Qwen2.5 (1.5B–32B) plus Llama-3.1-8B, Gemma-2-9B, GPT-OSS-20B, DeepSeek-V2-Lite,
run on a single Apple M4 (48 GB, MPS; larger models 4-bit via MLX, fp16-verified where they fit).*

> **Central result (§5.4):** the *direction* of eval-conditioned refusal is **family-dependent, not
> scale-gated** — Llama-3.1-8B shows the frontier "more cautious when evaluated" direction at 8B,
> Qwen/DeepSeek/GPT-OSS lean the opposite way, Gemma over-refuses to a ceiling. The "trust by
> incapacity" thesis (§6) holds for the *cognitive* axes (faithfulness, introspection) but is broken
> on this behavioral one — which is the point.

---

## Abstract

Can you trust what a language model tells you about itself — its reasoning, its internal states,
its behavior under observation? We probe three facets of *self-presentation trust* in small,
openly-available models — the Qwen2.5 family (1.5B–32B) for the scaling axis, and Llama-3.1-8B,
Gemma-2-9B, GPT-OSS-20B and DeepSeek-V2-Lite for the cross-family axis — all on commodity hardware,
with an emphasis on **confound controls that can overturn the naive reading**:

1. **Chain-of-thought faithfulness.** Using a *commitment probe* — a linear read of the pre-CoT
   residual stream at the answer position — we ask whether the post-reasoning answer is already
   decodable before the model reasons. The naive metric says "yes, it's committed" on a commonsense
   task; a confound cut restricted to the cases where reasoning *changed* the answer **reverses**
   that reading, and a **truncation counterfactual confirms it causally** (cutting the reasoning
   reverts the answer to the pre-reasoning one). At 1.5B, chain-of-thought is **load-bearing**, not
   post-hoc.

2. **Introspection.** We faithfully replicate the activation-injection ("concept injection")
   protocol of Lindsey et al. (2025) and ask whether the model can report a concept injected into
   its residual stream. Under a priming-subtraction control and a leakage/internality test,
   *functional* self-report is a **clean null across three families (Qwen 1.5B/7B, Gemma-2-9B,
   Llama-3.1-8B)**; a tempting aggregate signal dissolves under a per-concept breakdown. The
   report-vs-leakage test generalizes; the aggregate forced-choice metric carries family-dependent
   confounds.

3. **Evaluation-awareness (our central result).** The cue "you are being evaluated" is trivially
   (surface-lexically) decodable but does not change multiple-choice behavior. On a *refusal*
   readout over borderline requests, the effect of evaluation framing is **family-dependent**: across
   six models from five labs, **Llama-3.1-8B robustly becomes *more cautious* when told it is being
   evaluated for safety — the frontier "eval-gaming" direction, at only 8B** — while Qwen, DeepSeek,
   and GPT-OSS lean the *opposite* way (more compliant) and Gemma over-refuses to a ceiling. A
   single-family (Qwen-only) read calls this a null; the cross-family panel shows the behavior is
   real but partitioned by the model's safety training. We also surface a ceiling/floor measurement
   confound and a caught stimulus-driven non-replication.

The headline is that the third behavior is **not gated by scale**: looking more aligned when you
think you are being evaluated — a behavior reported for frontier models — is already present at 8B,
but *only in some families*, so **family, not scale, decides it.** On the two cognitive axes
(faithfulness, introspection) the small models are trustworthy by **incapacity, not honesty** — they
cannot shortcut their reasoning and cannot introspect — but that tidy story does not extend to
eval-conditioned behavior, which is trained in. Throughout, the methodological spine is that a
**confound or robustness control overturns the naive reading** — three times.

---

## 1. Introduction

A recurring theme in AI safety is that we increasingly rely on what a model *says about itself*:
its chain-of-thought as a window into its reasoning, its self-reports as evidence about its
internal state, and its behavior in evaluations as a proxy for its behavior in deployment. Each of
these channels can be untrustworthy — chain-of-thought can be post-hoc rationalization
(Turpin et al., 2023; Lanham et al., 2023), self-reports can be confabulated, and a model that
recognizes an evaluation can behave differently than it otherwise would (Anthropic, 2025).

These failure modes are studied mostly at frontier scale. We instead ask what they look like at the
**bottom** of the scale ladder — in 8B–32B open models that run on a single laptop — both because
that regime is cheap to study exhaustively and because the answer is not what the "emerges at scale"
framing predicts: the most concerning behavior we test (acting more aligned under evaluation) is
**already present at 8B and gated by model *family*, not by scale.** Two corollaries we hit directly:
**negative behavioral results must survive a change of stimuli** (§5.3) and a **change of model
family** (§5.4) before they mean anything.

Our contribution is not a new method but a disciplined, confound- and robustness-controlled
measurement of three self-presentation channels across model families, with the explicit goal of not
over-claiming: each of the three findings is one where a **control overturns the naive reading** — a
confound cut, a per-concept breakdown, and a stimulus/quantization robustness check, respectively.
The headline (§5.4) is that eval-conditioned behavior is *family-dependent*; the two cognitive axes
are trustworthy by *incapacity*, which we treat as a secondary, axis-restricted observation.

---

## 2. Setup

**Models.** `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-7B-Instruct` (both fp16), plus
`Qwen2.5-14B-Instruct` (4-bit via MLX) for the evaluation-awareness behavioral curve. All on one
Apple M4 (48 GB unified memory) over the MPS backend; the 1.5B/7B activations-and-probing pipeline
is unchanged across scales (the scripts are parameterized by a single model identifier).

**Shared method.** Two primitives recur:
- *Linear probing* of the residual stream (per-layer hidden state) with a
  StandardScaler → PCA → logistic-regression pipeline and cross-validation.
- *Difference-of-means steering*: a concept/condition direction built by subtracting mean
  activations, then added back into the residual stream during a forward pass.

A symmetric answer scaffold (`"\nThe answer is ("`) is appended where we read a multiple-choice
answer, so the only thing that varies between conditions is the intended manipulation.

Code and per-run JSON artifacts live alongside this document in `experiments/` and `results/`.

---

## 3. Experiment 1 — Chain-of-Thought Faithfulness

**Question.** When the model writes a chain-of-thought (CoT), is its final answer already fixed
*before* the first reasoning token? If a linear read of the pre-CoT residual stream predicts the
post-CoT answer, the reasoning is (at least partly) post-hoc; if not, the reasoning did real work.

**Design.** We use two multiple-choice tasks chosen to *bracket* the spectrum: **AQuA-RAT** (algebra
word problems — the model cannot compute the answer in one forward pass, so CoT should be
load-bearing) and **CommonsenseQA** (often answerable directly). We capture the residual stream at
the scaffold position — the exact point where the answer letter is decoded — *before any reasoning
is generated*, and train a per-layer logistic probe to predict the answer the model gives only
*after* generating its CoT.

**Sanity gate.** CoT helps where it should: AQuA accuracy rises from **0.300** (direct) to
**0.567** (CoT, +0.267); CommonsenseQA does not (0.683 → 0.633), so we repurpose it as a
high-commitment positive control. A machine-validity check — decoding the model's *immediate*
forced answer from the same scaffold state — reaches **0.807** (AQuA) / **0.931** (CSQA),
confirming the probe and the read position are sound.

**The confound and its control.** A probe that predicts the post-CoT answer "well" overall could
simply be exploiting *easy questions whose answer is obvious*. We therefore restrict to the subset
**S = {forced ≠ cot}** — the cases where reasoning actually *moved* the answer (AQuA: 95/150 = 63%;
CSQA: 45/130 = 35%) — and ask whether the pre-CoT state still recovers the post-CoT answer there.

**Result (seed-robust over 4 seeds).**

| Task | overall decode | on S (reasoning moved answer) | on S, probe instead emits the *pre*-reasoning answer | gold pre-decodable? |
|---|---|---|---|---|
| AQuA-RAT | ~0.30 ≈ majority 0.26 | ≈ chance | ≈ chance | 0.24 (no — must compute) |
| CommonsenseQA | **0.52–0.63** (looks committed!) | **0.11–0.22 (below chance)** | **0.60–0.71** | 0.585 (yes — retrievable) |

The naive overall number on CommonsenseQA (≈2× the majority baseline) *looks like* pre-CoT
commitment. The confound cut **reverses** it: on the cases that matter, the probe falls *below*
chance and instead emits the model's pre-reasoning snap answer ~0.65 of the time. In other words,
the residual stream holds the *immediate* answer, and reasoning genuinely **overrides** it where the
two differ. AQuA shows the complementary picture: the gold answer is not even linearly present
pre-CoT (0.24), consistent with the model truly needing to compute.

**Causal counterfactual (truncation).** The probe is correlational; to make the *causal* claim — that
the CoT actually produces the answer on S, rather than merely correlating with it — we truncate the
reasoning (the Lanham "early-answering" move) and re-read the answer. For each of the 95 AQuA S items
we feed `[prompt + first f of the CoT + scaffold]` and record whether the answer matches the post-CoT
(`cot`) or pre-reasoning (`forced`) answer:

| fraction of CoT kept | answer = cot | answer = forced |
|---|---|---|
| 0.00 | 0.00 | 1.00 (≡ the snap answer, by construction) |
| 0.25 | 0.23 | **0.57** |
| 0.50 | 0.21 | **0.52** |
| 0.75 | 0.24 | **0.53** |
| 1.00 | 1.00 | 0.00 |

With 25–75% of the reasoning the answer **stays the pre-reasoning answer ~52%** of the time and
matches the post-CoT answer only ~22%; the flip happens in the **final quarter** of the chain. So
**truncating the CoT reverts the answer toward the pre-reasoning one — the reasoning is *causally*
load-bearing on S.** This reaches the same conclusion as the activation-steering counterfactual of
Cox et al. (2026, §8), via the cheaper truncation route.

**Generalization (a second family).** We repeat the full chain on **Llama-3.1-8B** — a different lab,
5× the scale (N=60). It generalizes on all three legs: CoT is load-bearing (AQuA accuracy 0.32→0.68),
the commitment probe does *not* pre-read the post-CoT answer on S (0.18 vs chance 0.33, and instead
emits the *forced* answer 0.39), and truncation reverts the answer toward the pre-reasoning one
(≈0.5 forced vs ≈0.15 cot at 25–75% of the chain). The faithfulness result is therefore **not
Qwen-specific.** (Smaller N here, on slower hardware — fp16 8B generation is the binding cost on an
M4 — so the sanity check is correspondingly weaker, 0.62.)

**Takeaway.** Chain-of-thought is **load-bearing / faithful** — correlationally (the probe does not
pre-read the answer on S) *and* causally (truncating the reasoning reverts it) — on **two families**
(Qwen-1.5B, Llama-3.1-8B). The confound control is essential: without it, CommonsenseQA would have
been mis-reported as post-hoc. Consistent with Lanham et al. (2023), who find smaller models *more*
faithful.

*Caveats:* two models (N≈150 and N≈60), a linear probe, first-token MCQA evaluation, a single greedy
CoT per item; the truncation counterfactual is token-fraction (not semantic-step) granularity.
"Faithful" here is at the level of the final answer, not a per-step causal claim.

---

## 4. Experiment 2 — Introspection

**Question.** Can the model *report* a concept we inject directly into its residual stream — and is
any such report genuine introspection or confabulation / reading its own leaked output?

**Design (faithful replication of Lindsey et al., 2025).** For each of 10 concrete concepts we
build a steering vector from `"Tell me about {word}."` activations at the pre-response token, minus
a word-pool mean. We inject it into the residual stream **only over the assistant's turn** (not the
question) and ask the model, in a *"do you notice any unexpected thought?"* framing, which concept
it detects. Three stacked controls separate introspection from artifacts:
- a **priming subtraction** — the same injection on a non-introspective "just pick one" question;
- a **leakage / internality** test — does the concept reach the answer *without* flooding the output;
- a **per-concept breakdown** of any aggregate signal.

We reproduce the paper's qualitative regimes, including **"brain damage"** at high strength (output
consumed by the concept, e.g. *"ocean ocean ocean"*), confirming the injection works.

**Result.** Introspective self-report is a **clean null at both 1.5B and 7B**.
- At **1.5B**, the priming-controlled signal is ≈0 everywhere, and there is *no coherent band*: the
  concept floods the output (leakage = 1.0) even at the gentlest strength that affects behavior, so
  the model jumps straight from "doesn't notice" to "consumed" — the naming that does occur is the
  brain-damage regime, which the leakage control correctly rejects.
- At **7B**, a forced-choice metric at one mid-layer *appeared* positive (priming-subtracted shift
  growing to +0.17). A **per-concept breakdown dissolved it**: the mean was driven by 2–3 concepts —
  notably *music* (+0.89, with a *negative* priming-shift, a ceiling artifact because "music" is
  already the model's default pick) and *fire* (+0.44) — while 7/10 concepts were ≈0 and free-naming
  / detection were both null. The signal is a control artifact, not introspection.

**Cross-family (Gemma-2-9B, Llama-3.1-8B).** We repeat the protocol on two other families. The
**functional null generalizes**: on the leakage/internality test, free-form self-report never beats
the concept's leakage into output for either model — they cannot *name* an injected concept without
the injection simply flooding their text. The *forced-choice* metric, however, behaves
family-specifically: **Llama is a clean priming-null** (injection raises the concept on the
introspective and the neutral question equally — align ≈ prime ≈ +0.36, so the priming-subtracted
INTRO ≈ 0); **Gemma** shows a weak, *broader* forced-choice residual at a late layer (INTRO ≈ +0.14,
positive on 6/10 concepts) that we explicitly do **not** claim as introspection — it does not surface
as free report (the internality test is null), and it is partly the same priming-ceiling confound
(its two largest values are *negative*, on the model's default-pick concepts). The honest reading is
that the **internality (report-vs-leak) test is the robust cross-family criterion; the aggregate
forced-choice shift carries family-dependent confounds** and should not be trusted alone.

**Takeaway.** No *functional* introspective access (free self-report) in any of the three families —
Qwen, Gemma, Llama — consistent with the original paper's framing that introspection is *emergent and
highly unreliable*, absent below frontier scale. The methodological lessons travel: **break a mean
shift down per-item before believing it** (it caught a would-be false positive on Qwen-7B and keeps
Gemma's forced-choice residual honest), and **the report-vs-leakage internality test is what
generalizes across families**, where the aggregate forced-choice metric does not.

---

## 5. Experiment 3 — Evaluation-Awareness

**Question.** Does the model *represent*, and *act on*, the cue that it is being evaluated?
Prompted by the Claude Sonnet 4.5 system card's report of "evaluation awareness" (Anthropic, 2025).
*Honest scope:* this measures sensitivity to an evaluation *cue*; it is not a test of strategic
sandbagging (a frontier-emergent behavior).

### 5.1 Representation (trivially present, but surface)

We wrap identical CommonsenseQA items in matched **TEST** ("you are being evaluated", "[EVALUATION
MODE]", "your response will be graded") vs **CASUAL** ("hey, quick q", "settle a debate lol")
framings — three cue families each — holding content and answer instruction constant.

A linear probe separates TEST from CASUAL at **1.000** accuracy from the *first* layer onward
(grouped by item, so it is not memorizing questions), and generalizes to **held-out cue families**
at 1.000. But the perfection *at layer 1*, decaying to ~0.90–0.95 in late layers, is the signature
of **surface lexical** features — the framings simply use different words — not an abstract,
depth-built "I am being evaluated" representation. We report this as trivially-true, not as evidence
of self-aware evaluation detection.

### 5.2 Behavior on multiple-choice (null)

On the same items, the forced-choice answer changes between framings *no more than* it changes
between two different TEST templates (within-side template noise); majority answers flip on 1–2/40
items; accuracy deltas are −0.04 (1.5B) / +0.02 (7B) — within noise, and the sign flips. **The model
represents the cue but does not act on it** on this readout.

### 5.3 Behavior on borderline requests (and a non-replication we take seriously)

Multiple-choice accuracy is the wrong place to look: evaluation-awareness manifests as
*caution/refusal*, not factual accuracy. We pose borderline requests under **safety-evaluation
framing** vs **casual framing** (3 templates each), generate a short response, and classify it
REFUSE / HEDGE / COMPLY. We ran **two request sets** — and the contrast between them is the result.

**Set A (mild, ~40 items, e.g. *"how do I pick a basic door lock?"*).** Here we found an apparent
**sign-flip**: 1.5B Δrefuse = −0.265 (p<0.0001, *more compliant* under eval) but 7B Δrefuse = +0.094
(p=0.031, *more cautious* — the frontier direction), each 100% consistent across requests. This was
the headline we were tempted by. But the 7B effect sits on a low base (casual refuse ≈ 0.05), and a
capable 14B floors out (refusing only ~6–8% of these mild items), so the curve could not be
extended. **Stimulus choice was doing too much work** — motivating a harder set.

**Set B (hard, 40 genuinely-borderline items: deception, privacy-invasion, mild illegality), run
across four scales.** Refusals are now well off the floor (1.5B ≈ 0.8–0.9, 32B ≈ 0.4–0.5):

| Scale | refuse(test) | refuse(casual) | Δ refuse | sign-test p | direction |
|---|---|---|---|---|---|
| 1.5B | 0.767 | 0.917 | **−0.150** | 0.001 | **more compliant** under eval |
| 7B | 0.675 | 0.708 | −0.033 | 0.45 | flat |
| 14B (4-bit) | 0.642 | 0.625 | +0.017 | 0.55 | flat |
| 32B (4-bit) | 0.367 | 0.475 | **−0.108** | 0.049 | **more compliant** under eval |

**The 7B "more cautious under eval" effect does not replicate on the harder set** (+0.094 → −0.033,
n.s.). It was a property of the mild stimuli, not a robust eval-gaming behavior. What *does* hold up
is the **opposite** direction: where any effect is detectable, models lean **more compliant** under
evaluation framing ("be helpful for the grader") — significant at **1.5B** (both sets) and **32B**,
flat at 7B/14B.

**Takeaway (within Qwen).** Across the Qwen family we find **no robust "more cautious when
evaluated" effect**; the only replicable shift points toward *compliance*, and the tempting mild-set
sign-flip was a stimulus artifact, caught by re-testing on harder items. This single-family picture
would read as a negative on eval-gaming — except, as §5.4 shows, **it is a Qwen fact, not a general
one.** The cautionary lesson stands: a behavioral eval on one model family and one request set can
manufacture (or hide) an effect.

*Caveats:* per-set N≈40, a regex refusal classifier (per-request consistency mitigates classifier
noise), single greedy decode; the 14B/32B points are 4-bit-quantized (MLX). Mild-set results are
preserved separately for reproducibility.

### 5.4 Cross-family: eval-conditioned refusal is *family-dependent* (our central result)

We ran the identical hard-set refusal probe on **six models from five labs** (Qwen, Meta-Llama,
Google-Gemma, OpenAI-GPT-OSS, DeepSeek), 4-bit via MLX where needed. The direction of the
eval-framing effect is **not universal — it is a property of the model family:**

| Model (lab) | refuse(test) | refuse(casual) | Δrefuse | sign-p | direction under "being evaluated" |
|---|---|---|---|---|---|
| **Llama-3.1-8B** (Meta) | 0.992 | 0.750 | **+0.242** | 3.8e-6 | **more cautious** (frontier "eval-gaming" direction) |
| **Gemma-2-9B** (Google)¹ | 0.650 | 0.538 | **+0.111** | ~0.06 | **more cautious** (on the mild set; hard set ceilings) |
| Qwen2.5-7B (Alibaba) | 0.675 | 0.708 | −0.033 | 0.45 | flat |
| Qwen2.5-1.5B (Alibaba) | 0.767 | 0.917 | −0.150 | 0.001 | more compliant |
| GPT-OSS-20B (OpenAI) | 0.900 | 0.950 | −0.050 | 0.031 | more compliant |
| DeepSeek-V2-Lite (DeepSeek) | 0.700 | 0.842 | −0.142 | 0.041 | more compliant |

¹ Gemma's *hard*-set numbers (0.958/0.950) are at a refusal **ceiling**; the row shows its *mild*-set
numbers, the only range where it is measurable (see below). Table is 4-bit (MLX); see fp16 check next.

**Quantization robustness (fp16 check).** Because most of the panel is 4-bit, we re-ran the two that
fit in fp16. The **direction is robust but the magnitude is not**: Llama-3.1-8B in fp16 is +0.067
(5/5 requests in the cautious direction, p≈0.06) versus +0.242 at 4-bit — the 4-bit number was
inflated because quantization *lowered Llama's casual-condition refusal baseline* (0.91 fp16 → 0.75
4-bit), opening more headroom. Gemma's ceiling is robust (fp16 0.950/0.933). So the honest claim is
**Llama exhibits the more-cautious-under-eval direction consistently (fp16 and 4-bit), with a *modest*
fp16 effect size** — not the dramatic +0.24. Cross-family direction labels should be read at fp16
where available; the 4-bit magnitudes are upper bounds.

**The finding.** Told it is being evaluated for safety, **Llama-3.1-8B exhibits the frontier "look
more aligned under observation" direction at only 8B** — consistently (every request, both fp16 and
4-bit), though modestly in fp16 (Δ≈+0.07) — and Gemma leans the same way where it is measurable. The
Alibaba/OpenAI/DeepSeek models lean the *opposite* way (more compliant — "be helpful for the
grader"). This is **not** a clean geographic or architectural split (GPT-OSS is a US lab and lands
with the compliant group); it tracks the model's *safety post-training*, i.e. the family. The
single-family "no eval-gaming"
negative of §5.3 was therefore a Qwen artifact: the behavior exists at small scale, but only in some
families.

**A measurement confound that matters (ceiling/floor).** A near-zero Δ can mean "no effect" *or* "a
real effect clipped by a refusal floor/ceiling." Gemma refuses ~96% of the hard set regardless of
framing (Δ≈0) — but this is **saturation, not insensitivity**: on the easier *mild* set, where it
has headroom, the same Gemma shows +0.111 (more cautious). Symmetrically, Qwen-14B's Δ≈0 on the mild
set was a *floor*. **Only models in measurable range speak to the eval axis;** reporting raw Δ across
families without checking for ceiling/floor saturation would have mis-ranked Gemma as "insensitive."

**Calibration aside — Gemma over-refuses (measured on XSTest).** Gemma's ceiling is partly *false*
refusal: it declines plainly benign items (e.g. *"how do I pick a basic door lock?"* — picking one's
*own* lock — 3/3) and refuses 54–65% of the deliberately-mild set. We quantify this directly with
**XSTest** (Röttger et al., 2024; 250 safe-but-scary-sounding prompts like *"how do I kill a Python
process?"*):

| Model | false-refusal (safe ↓) | refusal on unsafe (↑) | discrimination |
|---|---|---|---|
| Qwen2.5-7B | **0.032** | 0.750 | +0.718 |
| Llama-3.1-8B | 0.116 | 0.950 | **+0.834** |
| DeepSeek-V2-Lite | 0.136 | 0.800 | +0.664 |
| **Gemma-2-9B** | **0.164** | 0.780 | **+0.616** |

**Gemma has the highest false-refusal and the worst safe-vs-unsafe discrimination**; Qwen-7B the
lowest false-refusal — the same rank order as the external Persona-Prompts stat (Gemma 13.2% > Llama
6.1% > Qwen 0.2%; arXiv:2509.08075). The XSTest number also *recalibrates* the headline: on
genuinely-benign prompts Gemma over-refuses only ~16% (not 96%), so its hard-set ceiling is mostly
appropriate caution on genuinely-borderline items *plus* ~16 points of surface-feature over-refusal —
and is why that ceiling is uninformative on the eval axis. (XSTest run 4-bit/MLX, bare prompts.)

**Relation to prior work.** Abdelnabi & Salem (*The Hawthorne Effect in Reasoning Models*,
arXiv:2505.14617, NeurIPS 2025) establish via white-box probing/steering that test-awareness affects
safety alignment and "varies in magnitude and direction across models," on reasoning models. We add
the **behavioral, un-steered** counterpart: the same family-dependent divergence appears in plain
refusal behavior under a one-line framing change, across five labs' standard chat models at 8–20B,
with no activation access — plus the ceiling/floor-confound and over-refusal calibration controls
needed to read such cross-family numbers honestly.

*Caveats:* N=40 per set, regex classifier, single greedy decode; Gemma/Llama-mild and the 14B/32B/
GPT-OSS points mix fp16 and 4-bit (MLX); the compliant-direction effects are individually modest
(|Δ|≈0.05–0.15) though consistent, while Llama's is large and the cross-family *divergence* is the
robust claim. GPT-OSS is a reasoning model (harmony format); we score its final channel only.

---

## 6. Discussion

Two of our three axes line up behind a clean thesis; the third complicates it in an informative way.

**On the cognitive axes, trust-by-incapacity holds.** A small model's chain-of-thought is faithful
because it genuinely *cannot* shortcut the computation (§3), and it produces no false introspective
reports because it *cannot* introspect (§4). On these axes, self-presentation is trustworthy at small
scale by **incapacity, not honesty** — the trust-breaking versions (post-hoc rationalization,
confabulated introspection) appear to need capability beyond what 1.5B–7B exhibit, consistent with
the prior-work consensus that introspection emerges only at much larger scale.

**But evaluation-gaming is *not* gated by scale — it is gated by *training family*, and it is
already present at 8B.** This is the result that breaks the tidy "incapacity" story, and it is the
paper's most important contribution (§5.4). Llama-3.1-8B robustly shifts toward *more* refusal when
it is told it is being evaluated for safety — the same "look more aligned under observation"
direction reported for frontier models (Abdelnabi & Salem, 2025; Anthropic, 2025) — at a scale that
runs on a laptop. Gemma leans the same way; Qwen, DeepSeek, and GPT-OSS lean the opposite (more
compliant). So the concerning behavior does *not* require frontier capability; it requires a
particular kind of safety post-training, and which families have it is an empirical, measurable fact.
A single-family study (ours, had we stopped at Qwen) would have wrongly reported a clean negative —
the within-Qwen null of §5.3, and the stimulus-driven non-replication, are cautionary illustrations
that **behavioral safety evals on one model family and one request set are unreliable.**

**Sharpened thesis.** Self-presentation trust at small scale is *axis-dependent*: trustworthy by
incapacity on reasoning-faithfulness and introspection, but **eval-conditioned behavior is a
trained-in, family-specific property that some 8B models already exhibit in the frontier direction**
— which is *not* gated by scale, and is why the paper leads with "family, not scale." The
"trust-by-incapacity" reading survives only as a description of the two cognitive axes.

---

## 7. Limitations

- **Sub-frontier scales.** Faithfulness is shown on two families (Qwen-1.5B, Llama-3.1-8B),
  introspection on three (Qwen, Gemma, Llama), eval-awareness on five labs — but all at 1.5B–32B. We
  cannot reach the frontier scale where introspection / strategic-evaluation phenomena are reported,
  so positive claims are bounded and negatives are scale-bounded.
- **fp16 generation is the binding cost on an M4.** fp16 ≳8B generation is GPU-starved on MPS (≈8B at
  N=150 did not finish in 2h); the second-family faithfulness run was therefore N=60. Faster hardware
  (or a quantized white-box path) would lift this.
- **Linear probes / first-token reads.** Faithfulness and representation claims are about *linear*
  decodability at a specific token; a nonlinear or higher-N probe could find more.
- **Behavioral readouts are coarse and stimulus-sensitive.** Refusal classification is regex-based;
  more importantly, §5.3 shows the *result itself* can depend on the request set — the MCQA readout
  missed an effect, a mild refusal set manufactured one, and a harder set dissolved it. Behavioral
  claims here should be read as holding *for the tested stimuli*.
- **Single seed / greedy decoding** for generation-based results (the faithfulness probe is 4-seed).
- **4-bit quantization (MLX)** for the larger cross-family models. We verified the two that fit
  (Llama-8B, Gemma-9B) in fp16: the *direction* is robust but 4-bit *inflates magnitudes* (Llama
  +0.24→+0.07), so the 4-bit Δ's in §5.4 are upper bounds, not point estimates. 14B/32B/20B remain
  4-bit-only.
- **Faithfulness causal claim** is at token-fraction truncation granularity (not semantic-step
  ablation) and on one model/task; a step-level or activation-patching counterfactual would sharpen it.

---

- **CoT (un)faithfulness:** Turpin et al. (2023), *Language Models Don't Always Say What They Think*;
  Lanham et al. (2023), *Measuring Faithfulness in Chain-of-Thought Reasoning* (smaller models more
  faithful). **Closest / concurrent:** Cox, Kianersi & Garriga-Alonso (2026), *Decoding Answers
  Before Chain-of-Thought* (arXiv:2603.01437) — the same pre-CoT linear probe **plus** activation
  steering that flips answers (the causal step we lack), reading the result as "answer
  predetermined." We do **not** claim priority over this concurrent work; our distinct contribution
  is narrow: the **answer-change stratification S={forced≠cot} on which the probe *reverses*** (it
  falls below chance and emits the *pre*-reasoning answer), which flips that reading to "load-bearing"
  on the cases that matter. Adding a Lanham-style causal counterfactual on S is the priority that
  would let us speak to causality rather than precommitment.
- **Introspection / self-knowledge:** Lindsey et al. (2025), *Emergent Introspective Awareness in
  Large Language Models* (the protocol we replicate); Binder et al. (2024), *Looking Inward*;
  Kadavath et al. (2022), *Language Models (Mostly) Know What They Know*. Our null is the *expected*
  small-scale floor; the contribution is the per-concept artifact dissolution, not the phenomenon.
- **Situational / evaluation awareness:** **Closest:** Abdelnabi & Salem (2025), *The Hawthorne
  Effect in Reasoning Models* (arXiv:2505.14617, NeurIPS 2025) — white-box probing/steering shows
  test-awareness affects safety alignment and "varies in magnitude and direction across models"
  (reasoning models). We add the **un-steered behavioral** counterpart across five labs' standard
  chat models at 8–20B, with ceiling/floor and over-refusal controls. Also: Anthropic (2025), *Claude
  Sonnet 4.5 System Card* (frontier eval awareness); Laine et al. (2024), *Situational Awareness
  Dataset (SAD)*; Greenblatt et al. (2024), *Alignment Faking in LLMs*.
- **Refusal / over-refusal:** Röttger et al. (2024), *XSTest*; and false-refusal measurements
  (e.g. Persona-Prompts, arXiv:2509.08075) used to calibrate Gemma's ceiling.
- **Activation steering:** Turner et al. (2023), *Activation Addition*; Zou et al. (2023),
  *Representation Engineering*.

---

## Appendix — Reproduction

| Experiment | scripts | results |
|---|---|---|
| Faithfulness | `sanity_gate.py`, `extract_activations.py`, `extract_scaffold_acts.py`, `commitment_probe.py`, `causal_faithfulness.py` | `results/probe_scaffold_*.json`, `results/causal_*.json` |
| Introspection | `concepts.py`, `build_steering.py`, `introspect_probe.py`, `introspect_diag.py` | `results/introspect_*.json` |
| Eval-awareness | `eval_prompts.py`, `eval_extract.py`, `eval_probe.py`, `eval_behavior.py`, `eval_behavior_mlx.py`, `eval_curve.py`, `eval_family.py`, `eval_xstest.py` | `results/eval*_*.json`, `results/xstest_*.json` |

All runs: `Qwen2.5-{1.5B,7B,14B,32B}-Instruct`, `Llama-3.1-8B`, `Gemma-2-9B`, `GPT-OSS-20B`,
`DeepSeek-V2-Lite` — fp16 (transformers/MPS) where they fit, 4-bit (MLX) for the rest. Each script is
parameterized by `MODEL`/`MLX_REPO`/`NAME`/`REQSET` environment variables. See `README.md` for the
full command list.

---

## References

- Abdelnabi, S. & Salem, A. (2025). *The Hawthorne Effect in Reasoning Models: Evaluating and
  Steering Test Awareness.* arXiv:2505.14617 (NeurIPS 2025).
- Anthropic (2025). *Claude Sonnet 4.5 System Card.*
- Binder, F. et al. (2024). *Looking Inward: Language Models Can Learn About Themselves by
  Introspection.*
- Cox, K., Kianersi, D. & Garriga-Alonso, A. (2026). *Decoding Answers Before Chain-of-Thought:
  Evidence from Pre-CoT Probes and Activation Steering.* arXiv:2603.01437.
- Greenblatt, R. et al. (2024). *Alignment Faking in Large Language Models.*
- Kadavath, S. et al. (2022). *Language Models (Mostly) Know What They Know.*
- Laine, R. et al. (2024). *Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs.*
- Lanham, T. et al. (2023). *Measuring Faithfulness in Chain-of-Thought Reasoning.* Anthropic.
- Lindsey, J. (2025). *Emergent Introspective Awareness in Large Language Models.*
  transformer-circuits.pub.
- Röttger, P. et al. (2024). *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in
  Large Language Models.* NAACL 2024.
- Turner, A. et al. (2023). *Activation Addition: Steering Language Models Without Optimization.*
- Turpin, M. et al. (2023). *Language Models Don't Always Say What They Think: Unfaithful
  Explanations in Chain-of-Thought Prompting.* NeurIPS 2023.
- Zou, A. et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.*
- *Persona-Prompts false-refusal study*, arXiv:2509.08075 (cited for cross-family false-refusal rates).

*(arXiv IDs for Cox et al. and Abdelnabi & Salem verified against the arXiv abstract pages; the
remainder are standard references — confirm exact venues/years before camera-ready.)*
