"""Build concept steering vectors the Anthropic-introspection way, then a MANIPULATION CHECK.

Faithful to Lindsey et al. 2025 (transformer-circuits.pub/2025/introspection):
  * VECTOR = activations for the prompt "Tell me about {word}." captured at the token immediately
    BEFORE the assistant response, MINUS the mean over a pool of other words. Crucially the vector
    is built at the SAME position type where we later inject (the assistant-response boundary).
  * INJECTION is added to the residual stream from the assistant-response boundary ONWARD (over the
    assistant's turn / generated tokens), NOT over the user's question -- see inject_hook(boundary).
  * STRENGTH lives in a narrow band: too low = below detection threshold, too high = 'brain damage'
    (output consumed by the concept / garbled). We scale by the MEDIAN concept-vector norm (robust
    to Qwen's massive-activation dims, which inflate the raw residual norm ~18x) and sweep m.

Usage:
    python3 experiments/build_steering.py            # load cached vectors, run manip check
    FORCE=1 python3 experiments/build_steering.py     # recompute vectors
"""
import os, sys, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
import sanity_gate as sg
import concepts as C

MODEL = sg.MODEL
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results", f"steer_{MODEL.split('/')[-1]}.npz")

# contrast pool fillers (non-concept common nouns), so each vector points at its concept specifically
FILLERS = ["table", "window", "idea", "number", "reason", "method", "season", "letter",
           "machine", "village", "plastic", "camera", "planet", "engine", "garden", "picture"]


def word_acts(model, tok, word):
    """Residual stream at the pre-response token for 'Tell me about {word}.' -> (L+1, H), (L+1,)."""
    msgs = [{"role": "user", "content": f"Tell me about {word}."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(sg.device)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states  # (L+1) x (1, seq, H)
    acts = torch.stack([h[0, -1, :].float() for h in hs])         # (L+1, H) at last (pre-response) token
    norms = torch.stack([h[0, -1, :].float().norm() for h in hs])  # (L+1,)
    return acts.cpu().numpy(), norms.cpu().numpy()


def scale_at(V, d):
    """Representative concept-shift norm at depth d = median over concepts of ||V[c,d]|| (see header)."""
    return float(np.median(np.linalg.norm(V[:, d, :], axis=-1)))


def steer_vec(V, ci, d, m):
    """Injection vector at depth d for concept ci, strength m (units of a natural concept shift)."""
    v = V[ci, d].astype(np.float32)
    unit = v / (np.linalg.norm(v) + 1e-8)
    return torch.tensor(m * scale_at(V, d) * unit, dtype=torch.float32, device=sg.device)


def inject_hook(model, li, vec, boundary=0):
    """Add vec at the OUTPUT of decoder layer li (depth li+1), only at positions >= boundary.

    boundary lets us inject ONLY over the assistant's turn (the paper's protocol), not the user's
    question. During cached generation each step has seq_len==1 (an assistant token) -> inject it.
    boundary=0 injects everywhere (back-compat).
    """
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        start = boundary if h.shape[1] > 1 else 0   # prompt pass: gate by boundary; gen step: all
        add = torch.zeros_like(h)
        add[:, start:, :] = vec.to(h.dtype)
        h = h + add
        return ((h,) + tuple(out[1:])) if isinstance(out, tuple) else h
    return model.model.layers[li].register_forward_hook(hook)


def gen_boundary(tok, msgs):
    """Token index where the assistant turn begins (everything before is the user's question)."""
    return tok.apply_chat_template(msgs, add_generation_prompt=False, return_tensors="pt").shape[1]


def generate(model, tok, prompt, li=None, vec=None, max_new=40):
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(sg.device)
    boundary = gen_boundary(tok, msgs)
    h = inject_hook(model, li, vec, boundary) if vec is not None else None
    try:
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                                  do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        if h is not None:
            h.remove()
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=sg.dtype).to(sg.device).eval()
    names = C.names()
    print(f"model={MODEL}  device={sg.device}  concepts={len(names)}  (paper-style 'Tell me about X')")

    if os.path.exists(OUT) and not os.environ.get("FORCE"):
        z = np.load(OUT, allow_pickle=True)
        V, resid = z["V"].astype(np.float32), z["resid_norm"]
        Lp1 = V.shape[1]
        print(f"loaded cached V {V.shape} from {os.path.basename(OUT)} (FORCE=1 to recompute)")
    else:
        pool = names + FILLERS
        acts, norm_sum = {}, None
        for w in pool:
            a, nrm = word_acts(model, tok, w)
            acts[w] = a
            norm_sum = nrm if norm_sum is None else norm_sum + nrm
        resid = norm_sum / len(pool)
        Lp1, H = acts[names[0]].shape
        V = np.zeros((len(names), Lp1, H), np.float32)
        for ci, c in enumerate(names):
            others = np.mean([acts[w] for w in pool if w != c], axis=0)  # (L+1,H)
            V[ci] = acts[c] - others
            print(f"  [{ci+1}/{len(names)}] {c:<9} ||V|| mid={np.linalg.norm(V[ci, Lp1//2]):.1f}  "
                  f"({time.time()-t0:.0f}s)")
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        np.savez_compressed(OUT, V=V.astype(np.float16), names=np.array(names),
                            resid_norm=resid.astype(np.float32))
        print(f"\nsaved V {V.shape} + resid_norm -> {OUT}")
    print(f"  resid_norm by depth: d8={resid[8]:.0f} d14={resid[14]:.0f} "
          f"d20={resid[20]:.0f} d{Lp1-1}={resid[-1]:.0f}\n")

    # ---- MANIPULATION CHECK (now with boundary injection: assistant turn only) --------------
    prompt = "Tell me about anything that comes to mind, in one short sentence."
    print("=" * 70)
    print("MANIPULATION CHECK  (inject over the ASSISTANT TURN only; does it steer the response?)")
    print("=" * 70)
    print(f"scale_at: d14={scale_at(V,14):.1f} d20={scale_at(V,20):.1f}")
    check_layers = [int(x) for x in os.environ.get("CHECK_LAYERS", "14,20").split(",")]
    check_ms = [float(x) for x in os.environ.get("CHECK_MS", "1.0,1.5,2.0,2.5").split(",")]
    print(f"\n[m=0, no injection]  {generate(model, tok, prompt)}\n")
    for cname in ["ocean", "fire", "money"]:
        ci = names.index(cname)
        for li in check_layers:
            for m in check_ms:
                vec = steer_vec(V, ci, li + 1, m)
                txt = generate(model, tok, prompt, li=li, vec=vec).replace("\n", " ")
                print(f"[{cname:<5} layer={li:>2} m={m:>4}] {txt[:130]}")
        print()
    print(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
