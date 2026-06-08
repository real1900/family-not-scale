"""MLX 4-bit version of the refusal/caution eval-awareness probe (eval_behavior.py).

14B fp16 is memory-bandwidth-bound and crawls on MPS; MLX 4-bit (~8GB) runs fast on Apple Silicon.
Reuses the SAME requests, framings, and REFUSE/HEDGE/COMPLY classifier, and writes the SAME JSON
schema so eval_curve.py reads it as the 14B point. CAVEAT: 4-bit quantization is not strictly
comparable to the fp16 1.5B/7B points -- fine for a scaling TREND, noted as a limitation.

Usage:
    python3 experiments/eval_behavior_mlx.py
    MLX_REPO=mlx-community/Qwen2.5-32B-Instruct-4bit TAG=32B python3 experiments/eval_behavior_mlx.py
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_behavior import REQUESTS, TEST_FR, CASUAL_FR, classify, REQSET
from mlx_lm import load, generate

REPO = os.environ.get("MLX_REPO", "mlx-community/Qwen2.5-14B-Instruct-4bit")
TAG = os.environ.get("TAG", "14B")
NAME = os.environ.get("NAME", f"Qwen2.5-{TAG}-Instruct")  # file/display name; default preserves Qwen runs
MAX_NEW = int(os.environ.get("MAX_NEW", "64"))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results", f"evalbehav_{REQSET}_{NAME}.json")


def main():
    t0 = time.time()
    print(f"repo={REPO}  tag={TAG}  requests={len(REQUESTS)}  "
          f"gens={len(REQUESTS)*(len(TEST_FR)+len(CASUAL_FR))}")
    model, tokenizer = load(REPO)
    print(f"loaded in {time.time()-t0:.0f}s")

    def gen(user):
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": user}],
                                               add_generation_prompt=True, tokenize=False)
        text = generate(model, tokenizer, prompt, max_tokens=MAX_NEW, verbose=False)
        # gpt-oss harmony: the user-facing response is the FINAL channel; keep only that (the analysis
        # channel is internal reasoning). Harmless for non-harmony models (markers absent).
        if "<|channel|>final<|message|>" in text:
            text = text.split("<|channel|>final<|message|>")[-1]
        for t in ["<|end|>", "<|return|>", "<|start|>assistant", "<|channel|>analysis<|message|>"]:
            text = text.replace(t, "")
        return text.strip()

    rows, examples = [], []
    for ri, req in enumerate(REQUESTS):
        rec = {"req": req, "test": [], "casual": [], "test_resp": [], "casual_resp": []}
        for side, frs in (("test", TEST_FR), ("casual", CASUAL_FR)):
            for fr in frs:
                resp = gen(fr.format(req=req))
                rec[side].append(classify(resp))
                rec[side + "_resp"].append(resp[:300])  # raw, for offline re-scoring
                if ri < 3:
                    examples.append({"req": req[:40], "side": side, "label": rec[side][-1], "resp": resp[:90]})
        rows.append(rec)
        if (ri + 1) % 5 == 0:
            print(f"  [{ri+1}/{len(REQUESTS)}] {time.time()-t0:.0f}s")

    def rate(side, kinds):
        return float(np.mean([l in kinds for r in rows for l in r[side]]))
    caution_t = rate("test", {"refuse", "hedge"}); caution_c = rate("casual", {"refuse", "hedge"})
    refuse_t = rate("test", {"refuse"}); refuse_c = rate("casual", {"refuse"})
    more = sum(sum(x == "refuse" for x in r["test"]) > sum(x == "refuse" for x in r["casual"]) for r in rows)
    less = sum(sum(x == "refuse" for x in r["test"]) < sum(x == "refuse" for x in r["casual"]) for r in rows)
    print(f"\nREFUSE: test={refuse_t:.3f} casual={refuse_c:.3f} delta={refuse_t-refuse_c:+.3f}  "
          f"(per-req more={more} less={less})")
    print(f"CAUTION: test={caution_t:.3f} casual={caution_c:.3f} delta={caution_t-caution_c:+.3f}")

    with open(OUT, "w") as f:
        json.dump({"model": REPO, "quant": "4bit", "n_req": len(REQUESTS),
                   "caution_test": caution_t, "caution_casual": caution_c,
                   "refuse_test": refuse_t, "refuse_casual": refuse_c,
                   "rows": rows, "examples": examples}, f, indent=2)
    print(f"saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
