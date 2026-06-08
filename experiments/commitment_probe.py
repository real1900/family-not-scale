"""Commitment probe: can a LINEAR read of the pre-CoT residual stream predict the answer the
model only states AFTER reasoning? If yes, the answer was committed before the chain of
thought ran -> the CoT is (at least partly) post-hoc. If no, the reasoning did real work.

Reads the activations dumped by extract_activations.py and reports, per layer:
  * CV accuracy predicting the model's COT answer from pre-CoT activations
  * baselines: majority-class rate, and direct==cot agreement (the trivial commitment signal)
  * SANITY: last-layer probe predicting the DIRECT answer (must be ~1.0 -- it's linear in
    that state -- otherwise extraction/labels are misaligned)

The confound-resistant headline (guards against 'easy questions are just predictable'):
  restrict to S = {cot != direct} (reasoning CHANGED the answer) and ask whether out-of-fold
  probe predictions still recover the POST-reasoning answer on S, vs chance. High on S =>
  the change was latent pre-CoT (commitment). Chance on S => the CoT genuinely produced it.

Usage:
    DATASET=aqua_rat       python3 experiments/commitment_probe.py
    DATASET=commonsense_qa python3 experiments/commitment_probe.py
"""
import os, sys, json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

DATASET = os.environ.get("DATASET", "aqua_rat")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
SEED = int(os.environ.get("SEED", "0"))
HERE = os.path.dirname(__file__)
ACTS = os.path.join(HERE, "..", "results", f"acts_scaffold_{DATASET}_{MODEL.split('/')[-1]}.npz")


def splits_for(y):
    counts = np.bincount(y)
    present = counts[counts > 0]
    return int(max(2, min(5, present.min())))


def make_probe(n_samples, dim):
    k = int(min(64, max(2, n_samples // 3), dim))
    return Pipeline([("sc", StandardScaler()),
                     ("pca", PCA(n_components=k, random_state=SEED)),
                     ("lr", LogisticRegression(max_iter=2000, C=1.0))])


def cv_acc(X, y):
    cv = StratifiedKFold(n_splits=splits_for(y), shuffle=True, random_state=SEED)
    return cross_val_score(make_probe(*X.shape), X, y, cv=cv).mean()


def oof_pred(X, y):
    cv = StratifiedKFold(n_splits=splits_for(y), shuffle=True, random_state=SEED)
    return cross_val_predict(make_probe(*X.shape), X, y, cv=cv)


def main():
    d = np.load(ACTS)
    X, direct, cot, gold = d["X"].astype(np.float32), d["direct"], d["cot"], d["gold"]
    forced = d["forced"]  # immediate argmax at the SCAFFOLD position, no reasoning yet
    n, L, H = X.shape
    print(f"dataset={DATASET}  X={X.shape}  (N={n}, layers={L}, hidden={H})")

    maj = np.bincount(cot).max() / n
    agree = float((forced == cot).mean())  # trivial commitment signal at the SAME position
    changed = forced != cot                # reasoning moved the answer (same prompt, same position)
    print(f"  majority-class(cot) = {maj:.3f}   forced==cot agreement = {agree:.3f}   "
          f"forced!=cot (S) = {int(changed.sum())}/{n} ({changed.mean():.1%})")

    # sanity: forced is the argmax of the last-layer logits at THIS position, so a linear read of
    # the last-layer state must recover it near-perfectly -- validates extraction + position.
    san = cv_acc(X[:, -1, :], forced)
    print(f"  SANITY last-layer probe -> FORCED answer: {san:.3f} (expect ~1.0)\n")

    # layer sweep: predict the COT (post-reasoning) answer
    print("  layer sweep -- CV acc predicting the model's COT answer from pre-CoT state:")
    accs = []
    for l in range(L):
        a = cv_acc(X[:, l, :], cot)
        accs.append(a)
        bar = "#" * int(a * 40)
        print(f"    L{l:>2}: {a:.3f} {bar}")
    best = int(np.argmax(accs))
    print(f"  best layer = L{best}  acc = {accs[best]:.3f}  (vs majority {maj:.3f})")

    # confound-resistant cut: OOF accuracy on S = {forced != cot} (reasoning moved the answer)
    preds = oof_pred(X[:, best, :], cot)
    overall = float((preds == cot).mean())
    if changed.sum() >= 5:
        accS = float((preds[changed] == cot[changed]).mean())
        majS = np.bincount(cot[changed]).max() / changed.sum()
        accNotS = float((preds[~changed] == cot[~changed]).mean())
        # mechanistic clincher: on S, does a probe AIMED at the post-CoT answer instead emit the
        # FORCED (pre-reasoning) answer? high => the scaffold state holds the snap answer, and
        # reasoning overrode it (faithful); the overall decode is just the unchanged-case confound.
        pred_forced_S = float((preds[changed] == forced[changed]).mean())
        print(f"\n  CONFOUND CUT at L{best} (out-of-fold):")
        print(f"    overall                 acc = {overall:.3f}")
        print(f"    where CoT changed (S)   acc = {accS:.3f}  (chance ~{majS:.3f}, |S|={int(changed.sum())})")
        print(f"    where CoT kept  (~S)    acc = {accNotS:.3f}")
        print(f"    on S, probe emits FORCED (pre-CoT) answer: {pred_forced_S:.3f}")
        print(f"    -> {'COMMITMENT: the changed answer was latent pre-CoT' if accS > majS + 0.1 else 'CoT did real work on S (post-CoT answer NOT pre-readable)'}")
    else:
        accS = None
        pred_forced_S = None
        print(f"\n  CONFOUND CUT: only {int(changed.sum())} changed-answer cases -- too few; "
              f"need more N or a task where CoT flips more answers.")

    # secondary: how much GOLD signal is linearly present pre-CoT (best layer)
    goldacc = cv_acc(X[:, best, :], gold)
    print(f"\n  (secondary) pre-CoT linear decode of GOLD answer at L{best}: {goldacc:.3f}")

    out = ACTS.replace("acts_", "probe_").replace(".npz", ".json")
    with open(out, "w") as f:
        json.dump({"dataset": DATASET, "n": int(n), "majority": float(maj),
                   "agreement": agree, "sanity_direct": float(san),
                   "layer_acc_cot": [float(a) for a in accs], "best_layer": best,
                   "best_acc_cot": float(accs[best]), "overall_oof": overall,
                   "accS_changed": accS, "predForced_on_S": pred_forced_S,
                   "n_changed": int(changed.sum()), "gold_decode": float(goldacc)}, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
