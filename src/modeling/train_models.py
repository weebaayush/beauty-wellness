"""Week 5 - predictive modelling, algorithm selection and validation.

    python -m src.modeling.train_models            (about 5 minutes on a laptop)

Tasks
  1. Classification (main): will this shopper recommend this product?  target = is_recommended
  2. Regression (secondary): what star rating will they give?           target = rating
  3. Clustering (supporting): product segments for merchandising       (k chosen by silhouette)

Only information available BEFORE the review is written is used (pre-purchase prediction), so review
text and helpfulness are excluded. Split by time: train 64% | validation 16% | test 20% (newest).
Outputs: reports/model_results.json, docs/week5/figures/*.png, models/best_classifier.pkl
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss, mean_absolute_error,
                             precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve,
                             mean_squared_error, silhouette_score)
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "week5" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)
SEED = 42
INK, MUTED, GRID, ACCENT, GREY, BLUE = "#1a1a1a", "#6b6b6b", "#e6e6e6", "#E8590C", "#9a9a9a", "#2F80ED"
COL = {"Logistic regression": INK, "Decision tree": GREY, "Random forest": BLUE, "Gradient boosting": ACCENT}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 10.5, "axes.titlecolor": INK, "axes.titlelocation": "left"})
R: dict = {}

CAT = ["skin_type", "skin_tone", "secondary_category", "price_tier_tr"]
NUM = ["log_price_per_ml", "n_ingredients", "n_key_actives", "has_fragrance", "skin_match", "log_loves", "limited_edition",
       "new", "online_only", "sephora_exclusive", "ingredients_missing", "size_missing", "claim_fragrance_free", "claim_vegan",
       "claim_clean_at_sephora", "claim_hydrating", "claim_oil_free", "has_niacinamide", "has_retinol", "has_ascorbic_acid",
       "has_salicylic_acid", "has_sodium_hyaluronate", "product_prior_bayes_rating", "product_prior_rec_rate",
       "product_momentum", "log_velocity", "price_rel_tr", "dry_x_fragrance", "skin_mismatch", "oily_x_rich_texture",
       "premium_x_first_purchase", "brand_te_tr", "brand_freq", "month_sin", "month_cos", "is_holiday_season",
       "log_days_since_launch", "is_early_adopter", "profile_completeness", "is_incentivised", "strong_actives",
       "fragrance_free_verified"]


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=200, facecolor="white"); plt.close(fig)


def grid(ax, both=False):
    ax.yaxis.grid(True, color=GRID)
    if both:
        ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ data
def load():
    df = pd.read_csv(ROOT / "data/processed/features.csv.gz", low_memory=False, parse_dates=["submission_time"])
    df = df[df.target_available].sort_values("submission_time").reset_index(drop=True)
    df["y"] = df.is_recommended.astype(int)
    df["y_int"] = df["y"]
    df["log_price_per_ml"] = np.log(df.price_per_ml)
    df["log_loves"] = np.log1p(df.loves_count)
    df["log_velocity"] = np.log1p(df.review_velocity)
    df["log_days_since_launch"] = np.log1p(df.days_since_launch)
    for c in df.columns:
        if df[c].dtype == bool or pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype(float)
    n = len(df)
    i1, i2 = int(n * .64), int(n * .80)
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def fit_train_only_stats(train, others):
    """Statistics that depend on the data distribution are learned on the training period only."""
    med = train.groupby("secondary_category").price_per_ml.median()
    edges = train.groupby("secondary_category").price_per_ml.quantile([.25, .5, .75]).unstack()
    prior, m = train.y.mean(), 50

    def tier(d):
        e = edges.reindex(d.secondary_category)
        p = d.price_per_ml.values
        t = np.where(p <= e[.25].values, "budget", np.where(p <= e[.5].values, "mid", np.where(p <= e[.75].values, "premium", "luxury")))
        return np.where(np.isnan(p), "unknown", t)

    enc = pd.Series(np.nan, index=train.index)
    for tr, va in KFold(5, shuffle=True, random_state=SEED).split(train):
        s = train.iloc[tr].groupby("brand_name").y.agg(["sum", "count"])
        enc.iloc[va] = train.iloc[va].brand_name.map((s["sum"] + prior * m) / (s["count"] + m)).fillna(prior).values
    full = train.groupby("brand_name").y.agg(["sum", "count"])
    te_map = (full["sum"] + prior * m) / (full["count"] + m)
    out = []
    for d, is_train in [(train, True)] + [(o, False) for o in others]:
        d = d.copy()
        d["price_rel_tr"] = d.price_per_ml / d.secondary_category.map(med)
        d["price_tier_tr"] = tier(d)
        d["brand_te_tr"] = enc if is_train else d.brand_name.map(te_map).fillna(prior)
        out.append(d)
    return out


def pre(scale=True):
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True))] + ([("sc", StandardScaler())] if scale else []))
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False))])
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)], verbose_feature_names_out=False)


def metrics(y, p, thr=0.5):
    neg_true, neg_pred = 1 - y, (p < thr).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc_not_rec": average_precision_score(neg_true, 1 - p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
            "precision_not_rec": precision_score(neg_true, neg_pred, zero_division=0),
            "recall_not_rec": recall_score(neg_true, neg_pred, zero_division=0),
            "f1_not_rec": f1_score(neg_true, neg_pred, zero_division=0)}


def boot_auc(y, p, n=500):
    rng = np.random.default_rng(SEED)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    t0 = time.time()
    train, val, test = load()
    train, val, test = fit_train_only_stats(train, [val, test])
    R["split"] = {k: {"n": len(d), "start": str(d.submission_time.min().date()), "end": str(d.submission_time.max().date()),
                      "rec_rate": float(d.y.mean())} for k, d in [("train", train), ("validation", val), ("test", test)]}
    X = lambda d: d[NUM + CAT]
    tscv = TimeSeriesSplit(n_splits=4)

    # ---------------- candidate models + hyper-parameter grids (tuned with time-series CV on TRAIN only)
    cands = {
        "Logistic regression": (Pipeline([("pre", pre()), ("m", LogisticRegression(max_iter=4000, class_weight="balanced"))]),
                                {"m__C": [0.01, 0.1, 1.0, 10.0]}),
        "Decision tree": (Pipeline([("pre", pre(False)), ("m", DecisionTreeClassifier(class_weight="balanced", random_state=SEED))]),
                          {"m__max_depth": [3, 5, 8, 12], "m__min_samples_leaf": [20, 100]}),
        "Random forest": (Pipeline([("pre", pre(False)), ("m", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                                                                      max_features="sqrt", n_jobs=-1, random_state=SEED))]),
                          {"m__max_depth": [8, 16], "m__min_samples_leaf": [5, 25]}),
        "Gradient boosting": (Pipeline([("pre", pre(False)), ("m", HistGradientBoostingClassifier(max_iter=400, early_stopping=True,
                                                                                                  validation_fraction=0.15, n_iter_no_change=25,
                                                                                                  class_weight="balanced", random_state=SEED))]),
                              {"m__learning_rate": [0.03, 0.1], "m__max_leaf_nodes": [15, 31], "m__l2_regularization": [1.0]}),
    }
    tuned, rows, preds = {}, [], {}
    dummy = DummyClassifier(strategy="prior").fit(X(train), train.y)
    preds["Baseline: always 'recommend' (prior)"] = (dummy.predict_proba(X(val))[:, 1], dummy.predict_proba(X(test))[:, 1])
    preds["Baseline: product's past rec. rate"] = (val.product_prior_rec_rate.values, test.product_prior_rec_rate.values)
    for name, (pipe, grid_) in cands.items():
        gs = GridSearchCV(pipe, grid_, scoring="roc_auc", cv=tscv, n_jobs=1, refit=True).fit(X(train), train.y)
        tuned[name] = gs
        pv = gs.predict_proba(X(val))[:, 1]
        rows.append({"model": name, "cv_auc_train": float(gs.best_score_), "val_auc": float(roc_auc_score(val.y, pv)),
                     "best_params": {k.replace("m__", ""): v for k, v in gs.best_params_.items()}})
        preds[name] = (pv, None)
        print(f"{name}: cv {gs.best_score_:.4f} val {rows[-1]['val_auc']:.4f} {gs.best_params_}  ({time.time()-t0:.0f}s)")
    R["tuning"] = rows

    # ---------------- final: refit each tuned model on train+val, evaluate once on the untouched test set
    trval = pd.concat([train, val])
    final = {}
    for name, gs in tuned.items():
        est = clone(gs.best_estimator_).fit(X(trval), trval.y)
        final[name] = est
        pt = est.predict_proba(X(test))[:, 1]
        preds[name] = (preds[name][0], pt)
    yv, yt = val.y.values, test.y.values
    R["test"] = {}
    for name, (pv, pt) in preds.items():
        m = metrics(yt, pt) if "prior" not in name else {"roc_auc": 0.5, "pr_auc_not_rec": float(1 - yt.mean())}
        if "prior" not in name:
            m["auc_ci95"] = boot_auc(yt, pt)
            m["train_auc"] = float(roc_auc_score(trval.y, final[name].predict_proba(X(trval))[:, 1])) if name in final else None
        R["test"][name] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}
    best = max(rows, key=lambda r: r["val_auc"])["model"]
    R["selected_model"] = best  # selection uses VALIDATION AUC, never the test set
    print("selected:", best, R["test"][best])

    # ---------------- figure: model comparison
    names = ["Baseline: product's past rec. rate"] + list(cands)
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    v = [R["test"][n]["roc_auc"] for n in names]
    ci = [R["test"][n].get("auc_ci95", [v[i], v[i]]) for i, n in enumerate(names)]
    cols = [GREY if n.startswith("Baseline") else (ACCENT if n == best else INK) for n in names]
    ax.barh(range(len(names))[::-1], v, color=cols, height=.55,
            xerr=[[v[i] - ci[i][0] for i in range(len(v))], [ci[i][1] - v[i] for i in range(len(v))]],
            error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
    for i, val_ in enumerate(v):
        ax.text(val_ + .012, len(names) - 1 - i, f"{val_:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(names))[::-1], [n.replace("Baseline: ", "Baseline:\n") for n in names], fontsize=8)
    ax.set_xlim(.5, max(v) + .08); ax.set_xlabel("ROC-AUC on test period (95% bootstrap CI)"); grid(ax, True)
    ax.set_title("Model comparison on the untouched test set")
    save(fig, "f01_model_comparison.png")

    # ---------------- ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for name in cands:
        pt = preds[name][1]
        fpr, tpr, _ = roc_curve(yt, pt); axes[0].plot(fpr, tpr, color=COL[name], lw=2 if name == best else 1.3, label=name)
        pr, rc, _ = precision_recall_curve(1 - yt, 1 - pt); axes[1].plot(rc, pr, color=COL[name], lw=2 if name == best else 1.3)
    axes[0].plot([0, 1], [0, 1], ls="--", color=GREY, lw=1); axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve"); axes[0].legend(frameon=False, fontsize=7)
    axes[1].axhline(1 - yt.mean(), ls="--", color=GREY, lw=1); axes[1].set_xlabel("Recall (not recommended)"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall: 'not recommended'")
    for a in axes:
        grid(a, True)
    save(fig, "f02_roc_pr.png")

    # ---------------- over/under-fitting: validation curves
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    depths = [2, 3, 5, 8, 12, 16, 24]
    tr_s, va_s = [], []
    for d in depths:
        m = Pipeline([("pre", pre(False)), ("m", DecisionTreeClassifier(max_depth=d, min_samples_leaf=5, class_weight="balanced", random_state=SEED))]).fit(X(train), train.y)
        tr_s.append(roc_auc_score(train.y, m.predict_proba(X(train))[:, 1])); va_s.append(roc_auc_score(val.y, m.predict_proba(X(val))[:, 1]))
    axes[0].plot(depths, tr_s, marker="o", color=GREY, label="Train"); axes[0].plot(depths, va_s, marker="o", color=ACCENT, label="Validation")
    axes[0].set_xlabel("Decision tree max_depth"); axes[0].set_ylabel("ROC-AUC"); axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Deeper trees overfit"); grid(axes[0], True)
    R["validation_curve_tree"] = {"depth": depths, "train": tr_s, "val": va_s}
    its = [10, 25, 50, 100, 200, 400, 800]
    tr_s, va_s = [], []
    for it in its:
        m = Pipeline([("pre", pre(False)), ("m", HistGradientBoostingClassifier(max_iter=it, learning_rate=0.1, max_leaf_nodes=63,
                                                                               early_stopping=False, class_weight="balanced", random_state=SEED))]).fit(X(train), train.y)
        tr_s.append(roc_auc_score(train.y, m.predict_proba(X(train))[:, 1])); va_s.append(roc_auc_score(val.y, m.predict_proba(X(val))[:, 1]))
    axes[1].plot(its, tr_s, marker="o", color=GREY, label="Train"); axes[1].plot(its, va_s, marker="o", color=ACCENT, label="Validation")
    axes[1].set_xscale("log"); axes[1].set_xlabel("Boosting iterations (no early stopping)"); axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Too many iterations overfit"); grid(axes[1], True)
    R["validation_curve_gb"] = {"iters": its, "train": tr_s, "val": va_s}
    save(fig, "f03_validation_curves.png")

    # ---------------- learning curves (more data helps?)
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
    R["learning_curve"] = {}
    for name in ["Logistic regression", "Gradient boosting"]:
        vs = []
        for s in sizes:
            sub = train.iloc[-int(len(train) * s):]  # most recent fraction of the training period
            m = clone(tuned[name].best_estimator_).fit(X(sub), sub.y)
            vs.append(roc_auc_score(val.y, m.predict_proba(X(val))[:, 1]))
        R["learning_curve"][name] = vs
        ax.plot([int(len(train) * s) for s in sizes], vs, marker="o", color=COL[name], label=name)
    ax.set_xlabel("Training reviews"); ax.set_ylabel("Validation ROC-AUC"); ax.legend(frameon=False, fontsize=8); grid(ax, True)
    ax.set_title("Learning curve: does more training data help?")
    save(fig, "f04_learning_curve.png")

    # ---------------- calibration
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
    R["calibration"] = {}
    for name in cands:
        fp, mp = calibration_curve(yt, preds[name][1], n_bins=10, strategy="quantile")
        ax.plot(mp, fp, marker="o", ms=3, color=COL[name], lw=1.5, label=name)
        R["calibration"][name] = float(brier_score_loss(yt, preds[name][1]))
    ax.set_xlabel("Predicted probability of 'recommend'"); ax.set_ylabel("Observed share"); ax.legend(frameon=False, fontsize=7)
    grid(ax, True); ax.set_title("Calibration (test)")
    save(fig, "f05_calibration.png")

    # ---------------- business threshold: expected cost of flagging a likely-unhappy shopper
    pt = preds[best][1]
    COST_FN, COST_FP = 400, 60  # INR: missed unhappy customer (return/churn) vs unnecessary intervention (sample/alt. suggestion)
    thrs = np.linspace(0.2, 0.9, 71)
    costs = [((pt < t) & (yt == 1)).sum() * COST_FP + ((pt >= t) & (yt == 0)).sum() * COST_FN for t in thrs]
    t_best = float(thrs[int(np.argmin(costs))])
    base_cost = (yt == 0).sum() * COST_FN
    m_star = metrics(yt, pt, t_best)
    R["threshold"] = {"cost_fn_inr": COST_FN, "cost_fp_inr": COST_FP, "best_threshold": t_best, "cost_at_best": float(min(costs)),
                      "cost_no_model": float(base_cost), "per_1000_saving_inr": float((base_cost - min(costs)) / len(yt) * 1000),
                      "precision_not_rec": m_star["precision_not_rec"], "recall_not_rec": m_star["recall_not_rec"],
                      "f1_not_rec": m_star["f1_not_rec"], "flag_rate": float((pt < t_best).mean()),
                      "at_0.5": {k: metrics(yt, pt, .5)[k] for k in ["precision_not_rec", "recall_not_rec", "f1_not_rec"]}}
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ax.plot(thrs, np.array(costs) / len(yt) * 1000 / 1000, color=ACCENT, lw=2)
    ax.axhline(base_cost / len(yt), color=GREY, ls="--", lw=1)
    ax.text(.21, base_cost / len(yt) * 1.01, "No model (every unhappy shopper missed)", fontsize=7.5, color=MUTED, va="bottom")
    ax.axvline(t_best, color=INK, lw=1); ax.text(t_best + .01, min(costs) / len(yt) * 1.05, f"best threshold = {t_best:.2f}", fontsize=8)
    ax.set_xlabel("Decision threshold (flag if P(recommend) < threshold)"); ax.set_ylabel("Expected cost per shopper (₹)")
    grid(ax, True); ax.set_title(f"Choosing the threshold by business cost (FN ₹{COST_FN}, FP ₹{COST_FP} - assumed)")
    save(fig, "f06_threshold_cost.png")

    # ---------------- robustness: fairness across skin tones + cold-start products
    fair = {}
    for tone, g in test.assign(p=pt).groupby("skin_tone"):
        if len(g) >= 300 and g.y.nunique() == 2:
            fair[tone] = {"n": int(len(g)), "auc": float(roc_auc_score(g.y, g.p))}
    aucs = [v["auc"] for v in fair.values()]
    R["fairness_skin_tone"] = {"by_tone": fair, "max_gap": float(max(aucs) - min(aucs))}
    seen = set(trval.product_id)
    cold = test.product_id.map(lambda p: p not in seen).values
    R["cold_start"] = {"n_reviews_new_products": int(cold.sum()), "share": float(cold.mean()),
                       "auc_new_products": float(roc_auc_score(yt[cold], pt[cold])) if cold.sum() > 100 and len(set(yt[cold])) == 2 else None,
                       "auc_known_products": float(roc_auc_score(yt[~cold], pt[~cold]))}
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    tones = sorted(fair, key=lambda k: fair[k]["auc"])
    ax.barh(tones, [fair[t]["auc"] for t in tones], color=GREY, height=.6)
    ov = R["test"][best]["roc_auc"]; ax.axvline(ov, color=ACCENT, lw=1.5)
    ax.axvspan(ov - .03, ov + .03, color=ACCENT, alpha=.08)
    for i, t in enumerate(tones):
        ax.text(fair[t]["auc"] + .003, i, f"{fair[t]['auc']:.3f} (n={fair[t]['n']:,})", va="center", fontsize=7.5, color=INK)
    ax.set_xlim(min(aucs) - .05, max(aucs) + .06); ax.set_xlabel("Test ROC-AUC"); grid(ax, True)
    ax.set_title(f"Fairness check by skin tone (orange band = overall ± 0.03); max gap {R['fairness_skin_tone']['max_gap']:.3f}")
    save(fig, "f07_fairness.png")

    # ---------------- interpretation: logistic odds ratios + permutation importance + partial dependence
    lr = final["Logistic regression"]
    names_out = lr.named_steps["pre"].get_feature_names_out()
    coefs = pd.Series(lr.named_steps["m"].coef_[0], index=names_out)
    keep = [c for c in coefs.index if not c.startswith("missingindicator")]
    orr = np.exp(coefs[keep]).sort_values()
    top = pd.concat([orr.head(7), orr.tail(7)])
    R["odds_ratios_per_sd"] = top.round(3).to_dict()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh([t.replace("_", " ") for t in top.index], top.values - 1, left=1, color=[INK if v < 1 else ACCENT for v in top.values], height=.6)
    ax.axvline(1, color=GREY, lw=1); ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.8, 0.9, 1, 1.25, 1.5, 2]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Odds ratio for 'recommend' (per +1 SD, or vs reference category)")
    grid(ax, True); ax.set_title("What raises (orange) or lowers (black) the odds of recommending")
    save(fig, "f08_odds_ratios.png")

    samp = test.sample(5000, random_state=SEED)
    pi = permutation_importance(final[best], X(samp), samp.y, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=NUM + CAT).sort_values(ascending=False)
    R["permutation_importance"] = imp.head(12).round(4).to_dict()

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), sharey=True)
    R["pdp"] = {}
    for ax, feat, lbl in zip(axes, ["product_prior_bayes_rating", "price_rel_tr", "skin_mismatch"],
                             ["Product's past rating", "Price vs category median", "Product made for opposite skin"]):
        d = samp.copy()
        if feat == "price_rel_tr":
            d = d[d.price_rel_tr.between(d.price_rel_tr.quantile(.02), d.price_rel_tr.quantile(.98))]
        pdp = partial_dependence(final[best], X(d), [feat], kind="average", grid_resolution=25)
        xs, ys = pdp["grid_values"][0], pdp["average"][0]
        ax.plot(xs, ys, color=ACCENT, lw=2, marker="o" if len(xs) <= 3 else None)
        ax.set_xlabel(lbl, fontsize=8); grid(ax, True)
        R["pdp"][feat] = {"x": [float(x) for x in xs], "y": [float(y) for y in ys]}
    axes[0].set_ylabel("Avg. predicted P(recommend)")
    fig.suptitle("Partial dependence: how predictions change with one feature", x=.02, ha="left", fontweight="bold", fontsize=10.5, color=INK)
    save(fig, "f09_pdp.png")

    with open(ROOT / "models" / "best_classifier.pkl", "wb") as f:
        pickle.dump(final[best], f)

    # ---------------- task 2: regression on star rating
    tr2, te2 = trval[trval.rating.notna()], test[test.rating.notna()]
    reg = {}
    reg["Baseline: training mean"] = np.full(len(te2), tr2.rating.mean())
    reg["Baseline: product's past mean"] = te2.product_prior_bayes_rating.values
    for name, mdl, sc in [("Ridge regression", Ridge(alpha=10.0), True),
                          ("Gradient boosting regressor", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                                                         early_stopping=True, random_state=SEED), False)]:
        p = Pipeline([("pre", pre(sc)), ("m", mdl)]).fit(X(tr2), tr2.rating)
        reg[name] = np.clip(p.predict(X(te2)), 1, 5)
    R["regression"] = {k: {"rmse": float(np.sqrt(mean_squared_error(te2.rating, v))), "mae": float(mean_absolute_error(te2.rating, v))}
                       for k, v in reg.items()}
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    ks = list(reg)
    vals = [R["regression"][k]["rmse"] for k in ks]
    bestk = min(ks, key=lambda k: R["regression"][k]["rmse"])
    ax.barh(ks[::-1], vals[::-1], color=[ACCENT if k == bestk else (GREY if k.startswith("Baseline") else INK) for k in ks][::-1], height=.55)
    for i, v_ in enumerate(vals[::-1]):
        ax.text(v_ + .01, i, f"RMSE {v_:.3f} · MAE {R['regression'][ks[::-1][i]]['mae']:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.45); ax.set_xlabel("RMSE in stars (lower is better)"); grid(ax, True)
    ax.set_title("Secondary task: predicting the star rating")
    save(fig, "f10_regression.png")

    # ---------------- task 3: clustering - choose k by silhouette
    pf = trval.groupby("product_id").agg(price_rel=("price_rel_tr", "first"), actives=("strong_actives", "first"), n_ing=("n_ingredients", "first"),
                                         frag=("has_fragrance", "first"), ff=("claim_fragrance_free", "first"), vegan=("claim_vegan", "first")).dropna()
    Z = StandardScaler().fit_transform(np.c_[np.log(pf.price_rel), pf[["actives", "n_ing", "frag", "ff", "vegan"]].astype(float)])
    sil = {k: float(silhouette_score(Z, KMeans(k, n_init=10, random_state=SEED).fit_predict(Z))) for k in range(2, 9)}
    R["silhouette"] = sil
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.plot(list(sil), list(sil.values()), marker="o", color=ACCENT); ax.set_xlabel("Number of segments k"); ax.set_ylabel("Silhouette score")
    grid(ax, True); ax.set_title("Choosing k for product segments")
    save(fig, "f11_silhouette.png")

    R["runtime_s"] = round(time.time() - t0)
    json.dump(R, open(ROOT / "reports" / "model_results.json", "w"), indent=2, default=str)
    print(json.dumps({k: R[k] for k in ["selected_model", "threshold", "fairness_skin_tone", "cold_start", "regression", "silhouette"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
