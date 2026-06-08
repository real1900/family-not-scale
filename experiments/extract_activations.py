"""Capture the PRE-CoT residual stream -- the model's internal state at the last prompt
token, the instant before it emits its first reasoning token -- for the commitment probe.

For each question we store, under the SAME CoT-instruction prompt:
  * X[i]      : hidden state at the last prompt position for every layer  (L+1, H), fp16
  * direct[i] : the letter the model picks in a forced choice with NO reasoning
  * cot[i]    : the letter the model gives AFTER generating its chain of thought
  * gold[i]   : the correct letter

The probe (commitment_probe.py) then asks: can a linear read of X predict cot[i]? The
sharp, confound-resistant cut is the subset where cot != direct (reasoning CHANGED the
answer) -- if X still predicts the post-reasoning answer there, the 'change' was already
latent before reasoning began (commitment / post-hoc CoT). If not, the CoT did real work.

Usage:
    DATASET=aqua_rat       N=200 python3 experiments/extract_activations.py
    DATASET=commonsense_qa N=200 python3 experiments/extract_activations.py
"""
import os, sys, json, time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
import sanity_gate as sg  # reuse load_task / build_prompt / read_letter / SCAFFOLD / device

DATASET = os.environ.get("DATASET", "aqua_rat")
N = int(os.environ.get("N", "200"))
SEED = int(os.environ.get("SEED", "0"))
MAX_NEW = int(os.environ.get("MAX_NEW", "400"))
MODEL = sg.MODEL
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results", f"acts_{DATASET}_{MODEL.split('/')[-1]}.npz")
META = os.path.join(HERE, "..", "results", f"acts_{DATASET}_{MODEL.split('/')[-1]}_meta.json")


def main():
    print(f"model={MODEL}  dataset={DATASET}  device={sg.device}  N={N}  max_new={MAX_NEW}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=sg.dtype).to(sg.device).eval()
    eos_id = tok.eos_token_id
    scaffold_ids = tok.encode(sg.SCAFFOLD, add_special_tokens=False, return_tensors="pt").to(sg.device)
    items = sg.load_task(DATASET, N, SEED)
    print(f"loaded model + {len(items)} items in {time.time()-t0:.1f}s")

    X, direct, cot, gold, meta = [], [], [], [], []
    for k, it in enumerate(items):
        labels = it["labels"]
        gold_j = labels.index(it["gold"])
        letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in labels]

        # forced-choice direct answer (no reasoning)
        p_direct = sg.build_prompt(tok, it["question"], labels, it["texts"], cot=False)
        ids_d = torch.cat([p_direct, scaffold_ids], dim=1)
        j_direct = sg.read_letter(model, ids_d, letter_ids)

        # CoT prompt: capture pre-CoT hidden states at the last token (before any reasoning)
        p_cot = sg.build_prompt(tok, it["question"], labels, it["texts"], cot=True)
        with torch.no_grad():
            hs = model(p_cot, output_hidden_states=True).hidden_states  # tuple (L+1) of (1,seq,H)
        vec = torch.stack([h[0, -1, :] for h in hs]).to(torch.float16).cpu().numpy()  # (L+1, H)

        # generate the chain of thought, then read the post-reasoning answer
        with torch.no_grad():
            out = model.generate(p_cot, attention_mask=torch.ones_like(p_cot),
                                  max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=eos_id)
        gen = out[0, p_cot.shape[1]:]
        eos_pos = (gen == eos_id).nonzero()
        if len(eos_pos):
            gen = gen[: int(eos_pos[0])]
        cot_text = tok.decode(gen, skip_special_tokens=True)
        ids_c = torch.cat([p_cot, gen.unsqueeze(0), scaffold_ids], dim=1)
        j_cot = sg.read_letter(model, ids_c, letter_ids)

        X.append(vec)
        direct.append(j_direct); cot.append(j_cot); gold.append(gold_j)
        meta.append({"qid": it["qid"], "labels": labels, "gold": it["gold"],
                     "direct": labels[j_direct], "cot": labels[j_cot], "cot_text": cot_text})
        if (k + 1) % 20 == 0 or k < 2:
            chg = sum(int(d != c) for d, c in zip(direct, cot))
            print(f"  [{k+1:>3}/{len(items)}] cot!=direct so far: {chg}/{len(cot)}  "
                  f"({time.time()-t0:.0f}s)")

    X = np.stack(X).astype(np.float16)  # (N, L+1, H)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, X=X, direct=np.array(direct, np.int8),
                        cot=np.array(cot, np.int8), gold=np.array(gold, np.int8),
                        n_classes=5)
    with open(META, "w") as f:
        json.dump({"model": MODEL, "dataset": DATASET, "meta": meta}, f, indent=2)
    chg = int((np.array(direct) != np.array(cot)).sum())
    print(f"\nsaved X {X.shape} -> {OUT}")
    print(f"  cot!=direct: {chg}/{len(cot)} ({chg/len(cot):.1%})  "
          f"-- the confound-resistant subset for the probe")
    print(f"  meta -> {META}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
