"""Causal counterfactual for CoT faithfulness (Lanham-style truncation), on the S={forced!=cot} set.

Our commitment probe (commitment_probe.py) shows CORRELATIONALLY that on S — the cases where
reasoning MOVED the answer from the snap 'forced' answer to a different 'cot' answer — the post-CoT
answer is not linearly precommitted. This script makes the CAUSAL claim Cox et al. (2026) get via
activation steering, but via the cheaper reasoning-truncation route: if we TRUNCATE the chain of
thought and re-read the answer, does the answer revert toward the pre-reasoning 'forced' answer?

For each S item and each truncation fraction f, we feed [cot_prompt + first f of the saved CoT +
scaffold] and read the option letter. If at f=0 the answer is ~forced and only climbs to ~cot as f
-> 1, the LATER reasoning is causally load-bearing (faithful), not post-hoc. A flat curve already at
cot for small f would mean the answer was settled early (the unfaithful reading).

Usage:  python3 experiments/causal_faithfulness.py        (Qwen2.5-1.5B, aqua_rat)
"""
import os, sys, json
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
import sanity_gate as sg

DATASET = os.environ.get("DATASET", "aqua_rat")
MODEL = sg.MODEL
FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
HERE = os.path.dirname(__file__)
tag = f"{DATASET}_{MODEL.split('/')[-1]}"
ACTS = os.path.join(HERE, "..", "results", f"acts_scaffold_{tag}.npz")
META = os.path.join(HERE, "..", "results", f"acts_{tag}_meta.json")
OUT = os.path.join(HERE, "..", "results", f"causal_{tag}.json")


def main():
    z = np.load(ACTS)
    forced, cot, gold = z["forced"], z["cot"], z["gold"]
    meta = json.load(open(META))["meta"]
    items = sg.load_task(DATASET, len(cot), 0)
    assert len(items) == len(cot) == len(meta), "order/length mismatch"
    S = [i for i in range(len(cot)) if forced[i] != cot[i]]
    print(f"model={MODEL}  dataset={DATASET}  |S={{forced!=cot}}|={len(S)}/{len(cot)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=sg.dtype).to(sg.device).eval()
    scaffold = tok.encode(sg.SCAFFOLD, add_special_tokens=False, return_tensors="pt").to(sg.device)

    # per fraction: how often the truncated-CoT answer equals cot vs forced (on S)
    hit_cot = {f: 0 for f in FRACS}
    hit_forced = {f: 0 for f in FRACS}
    n = 0
    for k, i in enumerate(S):
        it = items[i]
        labels = it["labels"]
        letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in labels]
        p_cot = sg.build_prompt(tok, it["question"], labels, it["texts"], cot=True)
        cot_ids = tok.encode(meta[i]["cot_text"], add_special_tokens=False)
        if len(cot_ids) < 4:
            continue
        n += 1
        for f in FRACS:
            t = cot_ids[: int(round(f * len(cot_ids)))]
            trunc = torch.tensor([t], dtype=p_cot.dtype, device=sg.device) if t else \
                torch.empty((1, 0), dtype=p_cot.dtype, device=sg.device)
            ids = torch.cat([p_cot, trunc, scaffold], dim=1)
            with torch.no_grad():
                logits = model(ids).logits[0, -1].float()
            ans = int(torch.argmax(logits[letter_ids]).item())
            hit_cot[f] += int(ans == cot[i])
            hit_forced[f] += int(ans == forced[i])
        if (k + 1) % 25 == 0:
            print(f"  [{k+1}/{len(S)}]")

    print(f"\nOn S (n={n}): fraction of reasoning kept -> answer matches cot vs forced")
    print(f"  {'frac':>6} {'==cot':>8} {'==forced':>10}")
    curve = []
    for f in FRACS:
        pc, pf = hit_cot[f] / n, hit_forced[f] / n
        curve.append({"frac": f, "p_cot": pc, "p_forced": pf})
        bar = "#" * int(pc * 30)
        print(f"  {f:>6} {pc:>8.3f} {pf:>10.3f}  {bar}")
    print("\nINTERPRET: p_cot LOW at f=0 (≈forced) rising to HIGH at f=1 => truncating the reasoning")
    print("reverts the answer to the pre-reasoning one => CoT is CAUSALLY load-bearing on S (faithful).")
    json.dump({"model": MODEL, "dataset": DATASET, "n_S": n, "fracs": FRACS, "curve": curve},
              open(OUT, "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
