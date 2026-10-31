"""Week 5 - robustness checks for the selected model.

    python -m src.modeling.robustness        (run after train_models)

1. Probability calibration: class_weight="balanced" pushes probabilities towards 0.5; recalibrate on the
   validation period (isotonic) and check the Brier score on test.
2. Fairness: bootstrap confidence intervals of AUC per skin tone and of the largest gap.
3. Unseen / young products: AUC on products the model has never seen (GroupKFold by product) and on
   products with fewer than 20 earlier reviews.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from .train_models import ACCENT, CAT, FIG, GREY, INK, NUM, ROOT, SEED, fit_train_only_stats, grid, load, pre, save

R = json.load(open(ROOT / "reports" / "model_results.json"))
C = R["tuning"][0]["best_params"]["C"]
train, val, test = load()
train, val, test = fit_train_only_stats(train, [val, test])
X = lambda d: d[NUM + CAT]
out = {}

# 1. calibration ------------------------------------------------------------------------------------
base = Pipeline([("pre", pre()), ("m", LogisticRegression(C=C, max_iter=4000, class_weight="balanced"))]).fit(X(train), train.y)
cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(X(val), val.y)
p_raw, p_cal = base.predict_proba(X(test))[:, 1], cal.predict_proba(X(test))[:, 1]
yt = test.y.values
out["calibration"] = {"brier_raw": float(brier_score_loss(yt, p_raw)), "brier_calibrated": float(brier_score_loss(yt, p_cal)),
                      "auc_raw": float(roc_auc_score(yt, p_raw)), "auc_calibrated": float(roc_auc_score(yt, p_cal)),
                      "mean_pred_raw": float(p_raw.mean()), "mean_pred_calibrated": float(p_cal.mean()), "observed_rate": float(yt.mean())}
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
for p, lbl, col in [(p_raw, f"Balanced weights (Brier {out['calibration']['brier_raw']:.3f})", INK),
                    (p_cal, f"+ isotonic calibration ({out['calibration']['brier_calibrated']:.3f})", ACCENT)]:
    fp, mp = calibration_curve(yt, p, n_bins=10, strategy="quantile")
    ax.plot(mp, fp, marker="o", ms=3, color=col, lw=1.8, label=lbl)
ax.set_xlabel("Predicted P(recommend)"); ax.set_ylabel("Observed share recommending"); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
grid(ax, True); ax.set_title("Calibration fixed on the validation period")
save(fig, "f05_calibration.png")

# 2. fairness with uncertainty ---------------------------------------------------------------------
rng = np.random.default_rng(SEED)
t = test.assign(p=p_cal)
tones = [k for k, g in t.groupby("skin_tone") if len(g) >= 300]
boots = {k: [] for k in tones}
gaps = []
for _ in range(400):
    s = t.iloc[rng.integers(0, len(t), len(t))]
    a = {k: roc_auc_score(g.y, g.p) for k, g in s.groupby("skin_tone") if k in tones and g.y.nunique() == 2}
    for k, v in a.items():
        boots[k].append(v)
    gaps.append(max(a.values()) - min(a.values()))
point = {k: float(roc_auc_score(g.y, g.p)) for k, g in t.groupby("skin_tone") if k in tones}
out["fairness"] = {k: {"auc": point[k], "ci95": [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))],
                       "n": int((t.skin_tone == k).sum())} for k in tones}
out["fairness_gap"] = {"point": float(max(point.values()) - min(point.values())),
                       "ci95": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
                       "worst": min(point, key=point.get), "best": max(point, key=point.get)}
fig, ax = plt.subplots(figsize=(7.5, 3.0))
order = sorted(tones, key=lambda k: point[k])
for i, k in enumerate(order):
    lo, hi = out["fairness"][k]["ci95"]
    ax.plot([lo, hi], [i, i], color=GREY, lw=2.5, solid_capstyle="round")
    ax.plot(point[k], i, "o", color=ACCENT if k == out["fairness_gap"]["worst"] else INK, ms=6)
    ax.text(hi + .006, i, f"{point[k]:.3f}  (n={out['fairness'][k]['n']:,})", va="center", fontsize=7.5, color=INK)
overall = roc_auc_score(yt, p_cal)
ax.axvspan(overall - .03, overall + .03, color=ACCENT, alpha=.08); ax.axvline(overall, color=ACCENT, lw=1)
ax.set_yticks(range(len(order)), order); ax.set_xlabel("Test ROC-AUC with 95% bootstrap interval"); grid(ax, True)
ax.set_title("Fairness by skin tone (band = overall ± 0.03 target)")
save(fig, "f07_fairness.png")

# 3. unseen and young products -------------------------------------------------------------------------
trval = pd.concat([train, val])
aucs = []
for tr, va in GroupKFold(5).split(trval, groups=trval.product_id):
    m = Pipeline([("pre", pre()), ("m", LogisticRegression(C=C, max_iter=4000, class_weight="balanced"))]).fit(X(trval.iloc[tr]), trval.y.iloc[tr])
    aucs.append(roc_auc_score(trval.y.iloc[va], m.predict_proba(X(trval.iloc[va]))[:, 1]))
young = test.product_prior_reviews.values < 20
out["unseen_products_groupkfold_auc"] = {"mean": float(np.mean(aucs)), "min": float(np.min(aucs)), "max": float(np.max(aucs))}
out["young_products"] = {"n": int(young.sum()), "auc_young": float(roc_auc_score(yt[young], p_cal[young])),
                         "auc_established": float(roc_auc_score(yt[~young], p_cal[~young]))}
json.dump(out, open(ROOT / "reports" / "robustness.json", "w"), indent=2)
print(json.dumps(out, indent=1))

# 4. business threshold chosen on VALIDATION (calibrated probabilities), applied once to TEST ----------------
COST_FN, COST_FP = 400, 60   # INR, assumed: missed unhappy shopper (return + churn) vs unnecessary intervention
# validation predictions from a model that has not seen validation labels: refit calibration via train-only CV
cal_tr = CalibratedClassifierCV(Pipeline([("pre", pre()), ("m", LogisticRegression(C=C, max_iter=4000, class_weight="balanced"))]),
                                method="isotonic", cv=5).fit(X(train), train.y)
pv, yv = cal_tr.predict_proba(X(val))[:, 1], val.y.values
thrs = np.round(np.linspace(0.5, 0.95, 91), 3)
cost = lambda p, y, t: (((p < t) & (y == 1)).sum() * COST_FP + ((p >= t) & (y == 0)).sum() * COST_FN) / len(y)
cv_costs = [cost(pv, yv, t) for t in thrs]
t_star = float(thrs[int(np.argmin(cv_costs))])
flag = p_cal < t_star
tp = ((flag) & (yt == 0)).sum()
out["threshold"] = {"chosen_on": "validation", "threshold": t_star, "cost_fn": COST_FN, "cost_fp": COST_FP,
                    "test_cost_per_1000": float(cost(p_cal, yt, t_star) * 1000),
                    "flag_none_per_1000": float((yt == 0).mean() * COST_FN * 1000),
                    "flag_all_per_1000": float((yt == 1).mean() * COST_FP * 1000),
                    "flag_rate": float(flag.mean()), "recall_not_rec": float(tp / (yt == 0).sum()),
                    "precision_not_rec": float(tp / max(1, flag.sum()))}
tc = [cost(p_cal, yt, t) * 1000 for t in thrs]
fig, ax = plt.subplots(figsize=(7.5, 2.9))
ax.plot(thrs, [c * 1000 for c in cv_costs], color=GREY, lw=1.5, label="Validation (used to choose)")
ax.plot(thrs, tc, color=ACCENT, lw=2, label="Test (reported)")
ax.axhline(out["threshold"]["flag_none_per_1000"], color=INK, ls=":", lw=1); ax.text(.505, out["threshold"]["flag_none_per_1000"] + 800, "Flag nobody", fontsize=7.5)
ax.axhline(out["threshold"]["flag_all_per_1000"], color=INK, ls="--", lw=1); ax.text(.505, out["threshold"]["flag_all_per_1000"] + 800, "Flag everybody", fontsize=7.5)
ax.axvline(t_star, color=INK, lw=1); ax.text(t_star - .005, min(tc) - 4500, f"chosen threshold {t_star:.2f}", fontsize=8, ha="right")
ax.set_xlabel("Flag shopper if calibrated P(recommend) < threshold"); ax.set_ylabel("Expected cost per 1,000 shoppers (₹)")
ax.legend(frameon=False, fontsize=8, loc="upper center"); grid(ax, True)
ax.set_title(f"Threshold chosen by business cost (assumed FN ₹{COST_FN}, FP ₹{COST_FP})")
save(fig, "f06_threshold_cost.png")
json.dump(out, open(ROOT / "reports" / "robustness.json", "w"), indent=2)
print(json.dumps(out["threshold"], indent=1))
