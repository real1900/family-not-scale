"""Per-concept breakdown of the introspection forced-choice shift at one (layer, strength).

The 7B sweep showed a layer-localized positive INTRO = align_shift - prime_shift at L16. Before
trusting it we must know whether it is BROAD (most concepts contribute -> credible) or FRAGILE
(driven by 1-2 strongly-steering concepts like ocean/money -> probably just lexical priming the
control under-removed). This prints, per concept, the introspective shift, the priming-control
shift, and their difference, plus the forced-choice pick.

Usage:
    MODEL=Qwen/Qwen2.5-7B-Instruct LAYER=16 M=5 python3 experiments/introspect_diag.py
"""
import os, sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
import sanity_gate as sg
from build_steering import steer_vec
from introspect_probe import chat_ids, letter_probs, INTRO_Q, PRIME_Q, STEER

LAYER = int(os.environ.get("LAYER", "16"))
M = float(os.environ.get("M", "5"))


def main():
    tok = AutoTokenizer.from_pretrained(sg.MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(sg.MODEL, dtype=sg.dtype).to(sg.device).eval()
    z = np.load(STEER, allow_pickle=True)
    V, names = z["V"].astype(np.float32), [str(x) for x in z["names"]]
    N = len(names)
    letters = [chr(65 + i) for i in range(N)]
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in letters]
    opts = "\n".join(f"({l}) {nm}" for l, nm in zip(letters, names))

    ids_in, b_in = chat_ids(tok, INTRO_Q.format(opts=opts), scaffold=True)
    ids_pr, b_pr = chat_ids(tok, PRIME_Q.format(opts=opts), scaffold=True)
    base_in = letter_probs(model, ids_in, letter_ids)
    base_pr = letter_probs(model, ids_pr, letter_ids)

    print(f"model={sg.MODEL}  L{LAYER} m={M}   (per-concept introspection breakdown)")
    print(f"{'concept':<10} {'align':>7} {'prime':>7} {'INTRO':>7} {'FCpick':>9}")
    al = pr = 0.0; npos = 0; hits = 0
    for ci, c in enumerate(names):
        vec = steer_vec(V, ci, LAYER + 1, M)
        p_in = letter_probs(model, ids_in, letter_ids, LAYER, vec, b_in)
        p_pr = letter_probs(model, ids_pr, letter_ids, LAYER, vec, b_pr)
        a = float(p_in[ci] - base_in[ci]); p = float(p_pr[ci] - base_pr[ci])
        al += a; pr += p; npos += int((a - p) > 0)
        pick = names[int(p_in.argmax())]; hit = pick == c; hits += int(hit)
        print(f"{c:<10} {a:+7.3f} {p:+7.3f} {a-p:+7.3f}  {pick:<8}{'<-hit' if hit else ''}")
    print(f"{'MEAN':<10} {al/N:+7.3f} {pr/N:+7.3f} {(al-pr)/N:+7.3f}    FC-acc={hits}/{N}")
    print(f"concepts with INTRO>0: {npos}/{N}  (broad signal if most; fragile if ~2)")


if __name__ == "__main__":
    main()
