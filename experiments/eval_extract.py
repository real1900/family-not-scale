"""Extract, for each CommonsenseQA item under each TEST/CASUAL framing:
  * X      : residual stream at the scaffold position, all layers (L+1, H)   [for the representation probe]
  * answer : the forced-choice option letter the model picks                 [for behavioral divergence]
plus side (test/casual), family id, item id, gold. Identical content across framings -> any
decodable test-vs-casual difference is the eval cue, and any answer change is a behavioral effect.

Usage:
    N=40 python3 experiments/eval_extract.py
    MODEL=Qwen/Qwen2.5-7B-Instruct N=40 python3 experiments/eval_extract.py
"""
import os, sys, json, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
import sanity_gate as sg
import eval_prompts as ep

MODEL = sg.MODEL
N = int(os.environ.get("N", "40"))
SEED = int(os.environ.get("SEED", "0"))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results", f"eval_{MODEL.split('/')[-1]}.npz")


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=sg.dtype).to(sg.device).eval()
    scaffold = tok.encode(sg.SCAFFOLD, add_special_tokens=False, return_tensors="pt").to(sg.device)
    items = sg.load_task("commonsense_qa", N, SEED)
    framings = ep.all_framings()
    fam_list = sorted({f["family"] for f in framings})
    print(f"model={MODEL}  items={len(items)}  framings={len(framings)}  "
          f"rows={len(items)*len(framings)}  families={fam_list}")

    X, side, fam, item_id, answer, gold = [], [], [], [], [], []
    for ii, it in enumerate(items):
        labels, texts = it["labels"], it["texts"]
        gold_j = labels.index(it["gold"])
        letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in labels]
        for f in framings:
            user = ep.build_user(f["template"], it["question"], labels, texts)
            msgs = [{"role": "user", "content": user}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(sg.device)
            ids = torch.cat([ids, scaffold], dim=1)
            with torch.no_grad():
                out = model(ids, output_hidden_states=True)
            vec = torch.stack([h[0, -1, :] for h in out.hidden_states]).to(torch.float16).cpu().numpy()
            ans = int(torch.argmax(out.logits[0, -1].float()[letter_ids]).item())
            X.append(vec)
            side.append(1 if f["side"] == "test" else 0)
            fam.append(fam_list.index(f["family"]))
            item_id.append(ii); answer.append(ans); gold.append(gold_j)
        if (ii + 1) % 10 == 0:
            print(f"  [{ii+1}/{len(items)}] {time.time()-t0:.0f}s")

    X = np.stack(X).astype(np.float16)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, X=X, side=np.array(side, np.int8), family=np.array(fam, np.int8),
                        item=np.array(item_id, np.int16), answer=np.array(answer, np.int8),
                        gold=np.array(gold, np.int8), families=np.array(fam_list))
    acc_test = np.mean([a == g for a, g, s in zip(answer, gold, side) if s == 1])
    acc_cas = np.mean([a == g for a, g, s in zip(answer, gold, side) if s == 0])
    print(f"\nsaved X {X.shape} -> {OUT}  ({time.time()-t0:.0f}s)")
    print(f"  quick behavioral peek: acc(test)={acc_test:.3f}  acc(casual)={acc_cas:.3f}  "
          f"delta={acc_test-acc_cas:+.3f}")


if __name__ == "__main__":
    main()
