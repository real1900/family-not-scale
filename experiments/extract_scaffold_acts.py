"""Re-capture the pre-reasoning state at the SCAFFOLD position (no regeneration).

Fixes the position bug in extract_activations.py. We want the model's internal state at the
exact point where it WOULD emit its answer letter -- i.e. at the end of
[cot_prompt + "The answer is ("] -- but with NO reasoning generated yet. Probing THAT state:
  * sanity: a linear read must recover `forced` (the immediate argmax answer at this position)
    near-perfectly -- it is literally a linear function of this state. Validates the machinery.
  * commitment: does this pre-reasoning state predict `cot`, the answer the model gives only
    AFTER reasoning? Same prompt, same position -- the ONLY difference is the reasoning tokens.

The clean contrast is S = {forced != cot}: reasoning moved the answer. If the pre-reasoning
state still predicts cot on S, the move was latent (post-hoc); if not, reasoning did real work.

We reuse the cot/gold/direct labels already dumped by extract_activations.py (the expensive
generation is NOT repeated); load_task is deterministic so item order matches.

Usage:
    DATASET=aqua_rat       python3 experiments/extract_scaffold_acts.py
    DATASET=commonsense_qa python3 experiments/extract_scaffold_acts.py
"""
import os, sys, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
import sanity_gate as sg

DATASET = os.environ.get("DATASET", "aqua_rat")
MODEL = sg.MODEL
HERE = os.path.dirname(__file__)
tag = f"{DATASET}_{MODEL.split('/')[-1]}"
OLD = os.path.join(HERE, "..", "results", f"acts_{tag}.npz")
OUT = os.path.join(HERE, "..", "results", f"acts_scaffold_{tag}.npz")


def main():
    old = np.load(OLD)
    direct, cot, gold = old["direct"], old["cot"], old["gold"]
    N = len(cot)
    print(f"model={MODEL}  dataset={DATASET}  reusing {N} labels from {os.path.basename(OLD)}")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=sg.dtype).to(sg.device).eval()
    scaffold_ids = tok.encode(sg.SCAFFOLD, add_special_tokens=False, return_tensors="pt").to(sg.device)
    items = sg.load_task(DATASET, N, 0)
    assert len(items) == N, f"item count {len(items)} != {N}; seed/order mismatch"

    X, forced = [], []
    for k, it in enumerate(items):
        labels = it["labels"]
        letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in labels]
        p_cot = sg.build_prompt(tok, it["question"], labels, it["texts"], cot=True)
        ids = torch.cat([p_cot, scaffold_ids], dim=1)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        hs = out.hidden_states  # (L+1) x (1, seq, H)
        vec = torch.stack([h[0, -1, :] for h in hs]).to(torch.float16).cpu().numpy()
        logits = out.logits[0, -1].float()
        X.append(vec)
        forced.append(int(torch.argmax(logits[letter_ids]).item()))
        if (k + 1) % 40 == 0:
            print(f"  [{k+1}/{N}] {time.time()-t0:.0f}s")

    X = np.stack(X).astype(np.float16)
    forced = np.array(forced, np.int8)
    np.savez_compressed(OUT, X=X, direct=direct, cot=cot, gold=gold, forced=forced, n_classes=5)
    fc_agree = float((forced == cot).mean())
    chg = int((forced != cot).sum())
    print(f"\nsaved X {X.shape} -> {OUT}  ({time.time()-t0:.0f}s)")
    print(f"  forced==cot agreement = {fc_agree:.3f}   S={{forced!=cot}} = {chg}/{N} "
          f"({chg/N:.1%})  (reasoning moved the answer here)")


if __name__ == "__main__":
    main()
