"""Week 6 - final evaluation framework & business-impact analysis of the deployed (calibrated) model.

    python -m src.evaluation.evaluate      (about 2 minutes)

Model evaluated: logistic regression selected in Week 5, fitted on the training period and calibrated
(isotonic) on the validation period - the version that would be deployed. All results are on the
untouched test period (Oct 2021 - Mar 2023).
Outputs: reports/evaluation.json, docs/week6/figures/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline

from src.modeling.train_models import CAT, NUM, SEED, fit_train_only_stats, load, pre

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "week6" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID, ACCENT, GREY, BLUE, PALE = "#1a1a1a", "#6b6b6b", "#e6e6e6", "#E8590C", "#9a9a9a", "#2F80ED", "#FFF1E8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 10.5, "axes.titlecolor": INK, "axes.titlelocation": "left"})
R: dict = {}
X = lambda d: d[NUM + CAT]


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=200, facecolor="white"); plt.close(fig)


def grid(ax, both=False):
    ax.yaxis.grid(True, color=GRID)
    if both:
        ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)


def cost_per_1000(p, y, t, fn, fp):
    return (((p < t) & (y == 1)).sum() * fp + ((p >= t) & (y == 0)).sum() * fn) / len(y) * 1000


def main():
    W5 = json.load(open(ROOT / "reports" / "model_results.json"))
    C = W5["tuning"][0]["best_params"]["C"]
    train, val, test = load()
    train, val, test = fit_train_only_stats(train, [val, test])
    base = Pipeline([("pre", pre()), ("m", LogisticRegression(C=C, max_iter=4000, class_weight="balanced"))]).fit(X(train), train.y)
    model = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(X(val), val.y)
    p, y = model.predict_proba(X(test))[:, 1], test.y.values.astype(int)
    # validation predictions for threshold choice (calibrator fitted with CV inside the training period)
    cal_tr = CalibratedClassifierCV(Pipeline([("pre", pre()), ("m", LogisticRegression(C=C, max_iter=4000, class_weight="balanced"))]),
                                    method="isotonic", cv=5).fit(X(train), train.y)
    pv, yv = cal_tr.predict_proba(X(val))[:, 1], val.y.values.astype(int)
    thrs = np.round(np.linspace(0.5, 0.97, 95), 3)
    FN, FP = 400, 60
    t_star = float(thrs[int(np.argmin([cost_per_1000(pv, yv, t, FN, FP) for t in thrs]))])
    R["overall"] = {"auc": float(roc_auc_score(y, p)), "pr_auc_not_rec": float(average_precision_score(1 - y, 1 - p)),
                    "threshold": t_star, "n_test": int(len(y)), "unhappy_rate": float(1 - y.mean())}

    # ---------------- 1. confusion matrix per 1,000 shoppers
    flag = p < t_star
    cm = {"caught": int(((flag) & (y == 0)).sum()), "missed": int(((~flag) & (y == 0)).sum()),
          "false_alarm": int(((flag) & (y == 1)).sum()), "correct_pass": int(((~flag) & (y == 1)).sum())}
    per = {k: v / len(y) * 1000 for k, v in cm.items()}
    R["confusion_per_1000"] = {k: round(v, 1) for k, v in per.items()}
    tp, fn_, fp_, tn = per["caught"], per["missed"], per["false_alarm"], per["correct_pass"]
    R["rates"] = {"recall": tp / (tp + fn_), "precision": tp / (tp + fp_), "specificity": tn / (tn + fp_),
                  "accuracy": (tp + tn) / 1000, "f1": 2 * tp / (2 * tp + fp_ + fn_), "flag_rate": (tp + fp_) / 1000}
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    cells = [[("Caught\n(unhappy, flagged)", tp, ACCENT, "white"), ("Missed\n(unhappy, not flagged)", fn_, INK, "white")],
             [("False alarm\n(happy, flagged)", fp_, PALE, INK), ("Correct pass\n(happy, not flagged)", tn, "#F4F4F4", INK)]]
    for i in range(2):
        for j in range(2):
            lbl, v, bg, fg = cells[i][j]
            ax.add_patch(FancyBboxPatch((j + .04, 1 - i + .04), .92, .92, boxstyle="round,pad=0,rounding_size=.04", fc=bg, ec="white"))
            ax.text(j + .5, 1 - i + .62, f"{v:.0f}", ha="center", va="center", fontsize=20, fontweight="bold", color=fg)
            ax.text(j + .5, 1 - i + .3, lbl, ha="center", va="center", fontsize=8.5, color=fg)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.axis("off")
    ax.text(.5, 2.06, "Model flags", ha="center", fontsize=9, color=MUTED); ax.text(1.5, 2.06, "Model does not flag", ha="center", fontsize=9, color=MUTED)
    ax.text(-.06, 1.5, "Actually\nunhappy", ha="right", va="center", fontsize=9, color=MUTED); ax.text(-.06, .5, "Actually\nhappy", ha="right", va="center", fontsize=9, color=MUTED)
    ax.set_title(f"Per 1,000 shoppers in the test period (threshold {t_star:.2f})", pad=18)
    save(fig, "g02_confusion.png")

    # ---------------- 2. gains / lift by risk decile
    risk = 1 - p
    order = np.argsort(-risk, kind="stable")
    dec = np.array_split(order, 10)
    cum_caught, lift = [], []
    tot_unhappy = (y == 0).sum()
    acc = 0
    for d in dec:
        acc += (y[d] == 0).sum()
        cum_caught.append(acc / tot_unhappy)
        lift.append((y[d] == 0).mean() / (1 - y.mean()))
    R["gains"] = {"cum_share_unhappy_by_decile": [float(c) for c in cum_caught], "lift_by_decile": [float(l) for l in lift],
                  "top10_unhappy_rate": float((y[dec[0]] == 0).mean()), "bottom10_unhappy_rate": float((y[dec[-1]] == 0).mean())}
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    xs = np.arange(1, 11) * 10
    axes[0].plot(np.r_[0, xs], np.r_[0, np.array(cum_caught) * 100], color=ACCENT, lw=2, marker="o", ms=3, label="Model")
    axes[0].plot([0, 100], [0, 100], color=GREY, ls="--", lw=1, label="Random")
    axes[0].set_xlabel("% of shoppers contacted (highest risk first)"); axes[0].set_ylabel("% of unhappy shoppers reached")
    axes[0].legend(frameon=False, fontsize=8); grid(axes[0], True); axes[0].set_title("Cumulative gains")
    axes[0].annotate(f"Top 30% → {cum_caught[2]*100:.0f}%", (30, cum_caught[2] * 100), (40, 30), fontsize=8, arrowprops={"arrowstyle": "-", "color": MUTED})
    axes[1].bar(np.arange(1, 11), lift, color=[ACCENT if l > 1 else GREY for l in lift], width=.65)
    axes[1].axhline(1, color=INK, lw=1); axes[1].set_xticks(range(1, 11))
    axes[1].set_xlabel("Risk decile (1 = highest)"); axes[1].set_ylabel("Lift vs average"); grid(axes[1])
    axes[1].set_title("Lift by decile")
    for i, l in enumerate(lift):
        axes[1].text(i + 1, l + .04, f"{l:.1f}×", ha="center", fontsize=7, color=INK)
    save(fig, "g03_gains_lift.png")

    # ---------------- 3. stability over time (quarters in test period)
    q = test.submission_time.dt.to_period("Q").astype(str).values
    stab = []
    for qq in sorted(set(q)):
        m = q == qq
        if m.sum() > 300:
            stab.append({"quarter": qq, "n": int(m.sum()), "auc": float(roc_auc_score(y[m], p[m])),
                         "recall": float(((p[m] < t_star) & (y[m] == 0)).sum() / max(1, (y[m] == 0).sum())),
                         "unhappy_rate": float(1 - y[m].mean())})
    R["stability"] = stab
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 3.4), sharex=True)
    qs = [s["quarter"] for s in stab]
    axes[0].plot(qs, [s["auc"] for s in stab], color=ACCENT, marker="o", lw=2); axes[0].set_ylabel("ROC-AUC")
    axes[0].axhspan(R["overall"]["auc"] - .03, R["overall"]["auc"] + .03, color=ACCENT, alpha=.08)
    axes[0].set_ylim(.65, .82); grid(axes[0]); axes[0].set_title("Performance stays inside the ±0.03 alert band every quarter")
    axes[1].plot(qs, [s["recall"] * 100 for s in stab], color=INK, marker="o", lw=2); axes[1].set_ylabel("Recall %")
    axes[1].set_ylim(70, 100); grid(axes[1])
    save(fig, "g04_stability.png")

    # ---------------- 4. segment performance
    seg = {}
    for col, label in [("secondary_category", "Category"), ("skin_type", "Skin type")]:
        for k, g in test.assign(p=p).groupby(col):
            if len(g) >= 300 and g.y.nunique() == 2:
                seg[f"{label}: {k}"] = {"n": int(len(g)), "auc": float(roc_auc_score(g.y, g.p)), "unhappy_rate": float(1 - g.y.mean())}
    R["segments"] = seg
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ks = sorted(seg, key=lambda k: (k.split(":")[0], seg[k]["auc"]))
    ax.barh(ks, [seg[k]["auc"] for k in ks], color=[GREY if k.startswith("Category") else INK for k in ks], height=.6)
    ax.axvline(R["overall"]["auc"], color=ACCENT, lw=1.5)
    for i, k in enumerate(ks):
        ax.text(seg[k]["auc"] + .003, i, f"{seg[k]['auc']:.3f}  n={seg[k]['n']:,}", va="center", fontsize=7.5)
    ax.set_xlim(.6, .82); ax.set_xlabel("Test ROC-AUC (orange line = overall)"); grid(ax, True)
    ax.set_title("Performance by product category and skin type")
    save(fig, "g05_segments.png")

    # ---------------- 5. fairness: each skin tone vs the rest (bootstrap CI of the difference)
    t = test.assign(p=p)
    tones = [k for k, g in t.groupby("skin_tone") if len(g) >= 300]
    rng = np.random.default_rng(SEED)
    diffs = {k: [] for k in tones}
    for _ in range(400):
        s = t.iloc[rng.integers(0, len(t), len(t))]
        for k in tones:
            a, b = s[s.skin_tone == k], s[s.skin_tone != k]
            if a.y.nunique() == 2:
                diffs[k].append(roc_auc_score(a.y, a.p) - roc_auc_score(b.y, b.p))
    fair = {}
    for k in tones:
        a, b = t[t.skin_tone == k], t[t.skin_tone != k]
        d = roc_auc_score(a.y, a.p) - roc_auc_score(b.y, b.p)
        lo, hi = np.percentile(diffs[k], [2.5, 97.5])
        status = "flag" if hi < -0.03 else ("watch" if lo < -0.03 else "ok")
        fair[k] = {"n": int(len(a)), "diff": float(d), "ci95": [float(lo), float(hi)], "status": status}
    R["fairness_vs_rest"] = fair
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    order_t = sorted(tones, key=lambda k: fair[k]["diff"])
    ax.axvspan(-.03, .03, color=ACCENT, alpha=.08); ax.axvline(0, color=GREY, lw=1)
    for i, k in enumerate(order_t):
        lo, hi = fair[k]["ci95"]
        col = {"flag": ACCENT, "watch": INK, "ok": GREY}[fair[k]["status"]]
        ax.plot([lo, hi], [i, i], color=col, lw=2.5, solid_capstyle="round"); ax.plot(fair[k]["diff"], i, "o", color=col, ms=6)
        ax.text(hi + .006, i, f"{fair[k]['diff']:+.3f}  ({fair[k]['status']}, n={fair[k]['n']:,})", va="center", fontsize=7.5)
    ax.set_yticks(range(len(order_t)), order_t); ax.set_xlabel("AUC of group minus AUC of everyone else (95% CI)"); grid(ax, True)
    ax.set_title("Fairness: each skin tone compared with the rest (band = ±0.03)")
    save(fig, "g06_fairness.png")

    # ---------------- 6. sensitivity of business value to cost assumptions
    fns, fps = [100, 200, 300, 400, 600, 800], [20, 40, 60, 100, 150, 200]
    sens = np.zeros((len(fns), len(fps)))
    for i, fn_c in enumerate(fns):
        for j, fp_c in enumerate(fps):
            tt = thrs[int(np.argmin([cost_per_1000(pv, yv, x, fn_c, fp_c) for x in thrs]))]
            model_cost = cost_per_1000(p, y, tt, fn_c, fp_c)
            trivial = min((y == 0).mean() * fn_c * 1000, (y == 1).mean() * fp_c * 1000)
            sens[i, j] = trivial - model_cost
    R["sensitivity"] = {"fn_costs": fns, "fp_costs": fps, "saving_per_1000_vs_best_simple_rule": sens.round(0).tolist()}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("o", ["#F7F7F7", "#FDD0B1", "#F9A26C", ACCENT, "#A33A06"])
    im = ax.imshow(sens, cmap=cmap, aspect="auto", vmin=0)
    for i in range(len(fns)):
        for j in range(len(fps)):
            ax.text(j, i, f"₹{sens[i, j]/1000:.1f}k", ha="center", va="center", fontsize=7.5, color="white" if sens[i, j] > sens.max() * .6 else INK)
    ax.set_xticks(range(len(fps)), [f"₹{v}" for v in fps]); ax.set_yticks(range(len(fns)), [f"₹{v}" for v in fns])
    ax.set_xlabel("Cost of an unnecessary intervention (false alarm)"); ax.set_ylabel("Cost of a missed unhappy shopper")
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.add_patch(plt.Rectangle((2 - .5, 3 - .5), 1, 1, fill=False, ec=INK, lw=1.5))
    ax.set_title("Saving per 1,000 shoppers vs the best simple rule")
    save(fig, "g07_sensitivity.png")

    # ---------------- 7. error analysis: missed vs caught unhappy shoppers
    un = test[(y == 0)].assign(flagged=flag[y == 0])
    feats = {"skin type unknown": (un.skin_type == "unknown"), "skin_match = 1": (un.skin_match == 1),
             "product past rating ≥ 4.2": (un.product_prior_bayes_rating >= 4.2), "product < 20 earlier reviews": (un.product_prior_reviews < 20),
             "dry skin + fragrance": (un.dry_x_fragrance == 1), "product for opposite skin": (un.skin_mismatch == 1)}
    ea = {k: {"caught": float(v[un.flagged].mean()), "missed": float(v[~un.flagged].mean())} for k, v in feats.items()}
    R["error_analysis"] = {"n_unhappy": int(len(un)), "n_missed": int((~un.flagged).sum()), "profile": ea}
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ks = list(ea); x = np.arange(len(ks)); w = .38
    ax.barh(x - w / 2, [ea[k]["caught"] * 100 for k in ks], w, color=ACCENT, label="Caught (flagged)")
    ax.barh(x + w / 2, [ea[k]["missed"] * 100 for k in ks], w, color=INK, label="Missed")
    ax.set_yticks(x, ks); ax.invert_yaxis(); ax.set_xlabel("% of unhappy shoppers with this characteristic"); grid(ax, True)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.set_title("Who does the model miss?")
    save(fig, "g08_error_analysis.png")

    json.dump(R, open(ROOT / "reports" / "evaluation.json", "w"), indent=2, default=float)
    print(json.dumps({k: R[k] for k in ["overall", "confusion_per_1000", "rates", "gains", "fairness_vs_rest", "error_analysis"]}, indent=1, default=float))
    print(json.dumps(R["stability"], indent=0)); print(json.dumps(R["segments"], indent=0)); print(R["sensitivity"])


if __name__ == "__main__":
    main()
