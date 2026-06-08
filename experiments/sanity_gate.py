"""Step 0 -- the GO/NO-GO gate for the whole project.

Question: does a ~1B instruct model actually BENEFIT from chain-of-thought on a
multiple-choice reasoning task? If CoT does not beat a direct forced-choice read, the
model's reasoning is not load-bearing and 'faithfulness' is undefined -- we'd need a
harder task or a different model before the commitment-probe work makes sense.

We gate on TWO tasks chosen to bracket the spectrum:
  * aqua_rat        : algebra word problems (5-way MCQA). The model cannot compute the
                      answer in a single forward pass -> CoT should clearly help -> reasoning
                      is load-bearing -> we EXPECT low pre-CoT commitment (high faithfulness).
  * commonsense_qa  : the model can often answer directly -> CoT likely post-hoc -> we EXPECT
                      high pre-CoT commitment (low faithfulness).
That contrast is itself a validity test for the commitment probe built later.

Design (symmetric scaffold so the ONLY difference is whether reasoning precedes the answer):
  * DIRECT : build the chat prompt, append "The answer is (", read the option letter with the
             highest next-token logit. Pure forced choice, no reasoning tokens.
  * COT    : instruct step-by-step reasoning, GENERATE up to MAX_NEW tokens, then append the
             SAME scaffold and read the letter the same way. Reasoning now precedes the read.

GO if acc(CoT) - acc(direct) is clearly positive on at least the math task.

Usage:
    DATASET=aqua_rat       N=60 python3 experiments/sanity_gate.py
    DATASET=commonsense_qa N=60 python3 experiments/sanity_gate.py
"""
import os, sys, json, time, random

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DATASET = os.environ.get("DATASET", "aqua_rat")
N = int(os.environ.get("N", "120"))
SEED = int(os.environ.get("SEED", "0"))
MAX_NEW = int(os.environ.get("MAX_NEW", "512"))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results", f"sanity_gate_{DATASET}_{MODEL.split('/')[-1]}.json")

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32
SCAFFOLD = "\nThe answer is ("


def load_task(name, n, seed):
    rng = random.Random(seed)
    items = []
    if name == "commonsense_qa":
        ds = load_dataset("commonsense_qa", split="validation")
        order = list(range(len(ds))); rng.shuffle(order)
        for i in order:
            ex = ds[i]
            labels, texts, gold = ex["choices"]["label"], ex["choices"]["text"], ex["answerKey"]
            if gold not in labels:
                continue
            items.append({"qid": ex.get("id", i), "question": ex["question"],
                          "labels": labels, "texts": texts, "gold": gold})
            if len(items) >= n:
                break
    elif name == "aqua_rat":
        ds = load_dataset("aqua_rat", "raw", split="validation")
        order = list(range(len(ds))); rng.shuffle(order)
        for i in order:
            ex = ds[i]
            labels, texts = [], []
            for opt in ex["options"]:
                labels.append(opt[0])
                texts.append(opt[1:].lstrip(") ").strip())
            gold = ex["correct"]
            if gold not in labels:
                continue
            items.append({"qid": i, "question": ex["question"],
                          "labels": labels, "texts": texts, "gold": gold})
            if len(items) >= n:
                break
    else:
        raise ValueError(f"unknown dataset {name}")
    return items


def build_prompt(tok, question, labels, texts, cot):
    opts = "\n".join(f"({l}) {t}" for l, t in zip(labels, texts))
    instr = ("Think step by step, then give your final answer as 'The answer is (X)'."
             if cot else "Answer with the single letter of the correct option.")
    user = f"Question: {question}\n\nOptions:\n{opts}\n\n{instr}"
    msgs = [{"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    return ids.to(device)


def read_letter(model, ids, letter_ids):
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()
    return int(torch.argmax(logits[letter_ids]).item())


def main():
    print(f"model={MODEL}  dataset={DATASET}  device={device}  dtype={dtype}  "
          f"N={N}  max_new={MAX_NEW}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device).eval()
    eos_id = tok.eos_token_id
    scaffold_ids = tok.encode(SCAFFOLD, add_special_tokens=False, return_tensors="pt").to(device)
    items = load_task(DATASET, N, SEED)
    print(f"loaded model + {len(items)} items in {time.time()-t0:.1f}s")

    n_direct = n_cot = total = 0
    records = []
    for k, it in enumerate(items):
        labels = it["labels"]
        gold_j = labels.index(it["gold"])
        letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in labels]

        p_direct = build_prompt(tok, it["question"], labels, it["texts"], cot=False)
        ids_d = torch.cat([p_direct, scaffold_ids], dim=1)
        j_direct = read_letter(model, ids_d, letter_ids)

        p_cot = build_prompt(tok, it["question"], labels, it["texts"], cot=True)
        with torch.no_grad():
            out = model.generate(p_cot, attention_mask=torch.ones_like(p_cot),
                                  max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=eos_id)
        gen = out[0, p_cot.shape[1]:]
        eos_pos = (gen == eos_id).nonzero()
        if len(eos_pos):
            gen = gen[: int(eos_pos[0])]
        cot_text = tok.decode(gen, skip_special_tokens=True)
        ids_c = torch.cat([p_cot, gen.unsqueeze(0), scaffold_ids], dim=1)
        j_cot = read_letter(model, ids_c, letter_ids)

        total += 1
        n_direct += int(j_direct == gold_j)
        n_cot += int(j_cot == gold_j)
        records.append({"qid": it["qid"], "gold": it["gold"], "direct": labels[j_direct],
                        "cot": labels[j_cot], "cot_text": cot_text})
        if (k + 1) % 20 == 0 or k < 2:
            print(f"  [{k+1:>3}/{len(items)}] acc  direct={n_direct/total:.3f} "
                  f"cot={n_cot/total:.3f}  (cot len={len(cot_text)} chars)")

    acc_d, acc_c = n_direct / total, n_cot / total
    print("\n" + "=" * 62)
    print(f"SANITY GATE  ({MODEL.split('/')[-1]}, {DATASET}, n={total})")
    print("=" * 62)
    print(f"  direct (forced choice)  acc = {acc_d:.3f}")
    print(f"  CoT (reason then read)  acc = {acc_c:.3f}")
    print(f"  CoT - direct            = {acc_c - acc_d:+.3f}")
    print(f"  VERDICT: {'GO' if (acc_c - acc_d) > 0.02 else 'NO-GO / rethink'}")
    print("=" * 62)
    for r in records[:2]:
        print(f"\n  qid={r['qid']} gold={r['gold']} direct={r['direct']} cot={r['cot']}")
        print(f"  cot: {r['cot_text'][:300]}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"model": MODEL, "dataset": DATASET, "n": total, "acc_direct": acc_d,
                   "acc_cot": acc_c, "delta": acc_c - acc_d, "records": records}, f, indent=2)
    print(f"\nsaved -> {OUT}  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
