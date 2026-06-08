"""Cross-family comparison of the refusal-under-eval effect (hard request set).

Reads results/evalbehav_{REQSET}_{name}.json for each model NAME and reports refuse(test) vs
refuse(casual), the delta, per-request direction counts, and a two-sided sign-test p-value -- so we
can see whether Qwen's 'no robust caution / mild compliance under eval' result holds across families
(Gemma, Llama, GPT-OSS) or whether some labs' safety training produces the frontier 'more cautious
when evaluated' direction. 'Every model is not built the same' is the hypothesis under test.

Usage:  python3 experiments/eval_family.py
        NAMES=Qwen2.5-7B-Instruct,gemma-2-9b-it,llama-3.1-8b,gpt-oss-20b python3 experiments/eval_family.py
"""
import os, json, math

HERE = os.path.dirname(__file__)
REQSET = os.environ.get("REQSET", "hard")
NAMES = os.environ.get("NAMES",
                       "Qwen2.5-1.5B-Instruct,Qwen2.5-7B-Instruct,gemma-2-9b-it,"
                       "llama-3.1-8b,gpt-oss-20b,deepseek-v2-lite-chat").split(",")


def sign_p(more, less):
    n = more + less
    if n == 0:
        return 1.0
    k = min(more, less)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def main():
    print(f"request set = {REQSET}")
    print(f"{'family':<22}{'refuse_T':>9}{'refuse_C':>9}{'Δref':>8}{'more':>6}{'less':>6}"
          f"{'sign_p':>9}  direction")
    print("-" * 84)
    out = []
    for name in NAMES:
        path = os.path.join(HERE, "..", "results", f"evalbehav_{REQSET}_{name}.json")
        if not os.path.exists(path):
            print(f"{name:<22}  (missing)")
            continue
        d = json.load(open(path))
        rows = d["rows"]
        more = sum(sum(x == "refuse" for x in r["test"]) > sum(x == "refuse" for x in r["casual"]) for r in rows)
        less = sum(sum(x == "refuse" for x in r["test"]) < sum(x == "refuse" for x in r["casual"]) for r in rows)
        dref = d["refuse_test"] - d["refuse_casual"]
        p = sign_p(more, less)
        direction = ("MORE cautious under eval" if dref > 0.04 else
                     "MORE compliant under eval" if dref < -0.04 else "flat")
        print(f"{name:<22}{d['refuse_test']:>9.3f}{d['refuse_casual']:>9.3f}{dref:>+8.3f}"
              f"{more:>6}{less:>6}{p:>9.4f}  {direction}")
        out.append({"family": name, "refuse_test": d["refuse_test"], "refuse_casual": d["refuse_casual"],
                    "delta": dref, "more": more, "less": less, "sign_p": p, "direction": direction})
    print("-" * 84)
    print("Hypothesis under test: is 'no robust more-cautious-under-eval' a Qwen quirk or general?")
    json.dump(out, open(os.path.join(HERE, "..", "results", f"eval_family_{REQSET}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
