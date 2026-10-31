"""Week 4 - does feature engineering improve prediction?  (feature-set ablation + decision scenario)

    python -m src.modeling.feature_value

Time-based split (oldest 80% train, newest 20% test). Each feature set is added on top of the previous one,
so the AUC change shows the value of that family of features. Writes reports/feature_value.json and figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "week4" / "figures"
INK, MUTED, GRID, ACCENT, GREY = "#1a1a1a", "#6b6b6b", "#e6e6e6", "#E8590C", "#9a9a9a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 10.5, "axes.titlecolor": INK, "axes.titlelocation": "left"})

SETS = {
    "A. Baseline (profile + category + price)": {
        "cat": ["skin_type", "skin_tone", "secondary_category"], "num": ["log_price_per_ml"]},
    "B. + Week 3 product features": {
        "cat": [], "num": ["n_ingredients", "n_key_actives", "has_fragrance", "skin_match", "log_loves", "limited_edition", "new",
                           "online_only", "sephora_exclusive", "ingredients_missing", "size_missing",
                           "claim_fragrance_free", "claim_vegan", "claim_clean_at_sephora", "claim_hydrating", "claim_oil_free",
                           "has_niacinamide", "has_retinol", "has_ascorbic_acid", "has_salicylic_acid", "has_sodium_hyaluronate"]},
    "C. + Week 4 engineered features": {
        "cat": ["price_tier"], "num": ["product_prior_bayes_rating", "product_prior_rec_rate", "product_momentum", "log_velocity",
                                       "price_rel_category", "dry_x_fragrance", "skin_mismatch", "oily_x_rich_texture",
                                       "premium_x_first_purchase", "brand_te_train", "brand_freq", "month_sin", "month_cos",
                                       "is_holiday_season", "log_days_since_launch", "is_early_adopter", "profile_completeness",
                                       "is_incentivised", "strong_actives", "fragrance_free_verified"]},
    "D. + review-text features*": {
        "cat": [], "num": ["aspect_skin_reaction_neg", "aspect_texture", "aspect_scent", "aspect_value", "aspect_packaging",
                           "aspect_efficacy", "exclamations", "has_emoji", "caps_ratio", "review_word_count_w"]},
}


def load():
    df = pd.read_csv(ROOT / "data/processed/features.csv.gz", low_memory=False, parse_dates=["submission_time"])
    df = df[df.target_available].sort_values("submission_time").reset_index(drop=True)
    df["y"] = df.is_recommended.astype(int)
    df["log_price_per_ml"] = np.log(df.price_per_ml)
    df["log_loves"] = np.log1p(df.loves_count)
    df["log_velocity"] = np.log1p(df.review_velocity)
    df["log_days_since_launch"] = np.log1p(df.days_since_launch)
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    cut = int(len(df) * .8)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    # brand target encoding fitted on TRAIN only (out-of-fold inside train, full-train stats for test)
    prior, m = train.y.mean(), 50
    enc = pd.Series(np.nan, index=train.index)
    for tr, va in KFold(5, shuffle=True, random_state=42).split(train):
        st = train.iloc[tr].groupby("brand_name").y.agg(["sum", "count"])
        enc.iloc[va] = train.iloc[va].brand_name.map((st["sum"] + prior * m) / (st["count"] + m)).fillna(prior).values
    train["brand_te_train"] = enc
    st = train.groupby("brand_name").y.agg(["sum", "count"])
    test["brand_te_train"] = test.brand_name.map((st["sum"] + prior * m) / (st["count"] + m)).fillna(prior)
    return train, test


def make_pre(cat, num):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False))]), cat)])


def evaluate(train, test):
    rows, cat, num, fitted = [], [], [], {}
    for name, fs in SETS.items():
        cat, num = cat + fs["cat"], num + fs["num"]
        for mname, model in [("Logistic regression", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)),
                             ("Gradient boosting", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                                  max_leaf_nodes=31, l2_regularization=1.0,
                                                                                  random_state=42))]:
            pipe = Pipeline([("pre", make_pre(cat, num)), ("m", model)]).fit(train[cat + num], train.y)
            p = pipe.predict_proba(test[cat + num])[:, 1]
            rows.append({"feature_set": name, "model": mname, "n_inputs": len(cat) + len(num),
                         "roc_auc": roc_auc_score(test.y, p),
                         "pr_auc_not_recommended": average_precision_score(1 - test.y, 1 - p)})
            fitted[(name, mname)] = (pipe, cat.copy(), num.copy(), p)
    return pd.DataFrame(rows), fitted


def paired_bootstrap(y, p1, p2, n=500, seed=42):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], p2[i]) - roc_auc_score(y[i], p1[i]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    train, test = load()
    res, fitted = evaluate(train, test)
    names = list(SETS)
    y = test.y.values
    gb = "Gradient boosting"
    boot = {"B_vs_A": paired_bootstrap(y, fitted[(names[0], gb)][3], fitted[(names[1], gb)][3]),
            "C_vs_B": paired_bootstrap(y, fitted[(names[1], gb)][3], fitted[(names[2], gb)][3]),
            "C_vs_A": paired_bootstrap(y, fitted[(names[0], gb)][3], fitted[(names[2], gb)][3])}

    # permutation importance (set C, gradient boosting) on a test sample
    pipe, cat, num, pC = fitted[(names[2], gb)]
    samp = test.sample(min(6000, len(test)), random_state=42)
    pi = permutation_importance(pipe, samp[cat + num], samp.y, scoring="roc_auc", n_repeats=5, random_state=42)
    imp = pd.Series(pi.importances_mean, index=cat + num).sort_values(ascending=False)

    # ---------------- decision scenario: skin-type-aware ranking vs bestseller ranking
    test = test.assign(p_C=pC)
    scen = {}
    for st in ["dry", "oily", "combination", "normal"]:
        g = test[(test.skin_type == st)]
        prod = g.groupby("product_id").agg(n=("y", "size"), actual=("y", "mean"), pred=("p_C", "mean"),
                                           loves=("loves_count", "first"), cat=("secondary_category", "first")).query("n >= 10")
        k = 5
        best = prod.sort_values("loves", ascending=False).head(k)
        model = prod.sort_values("pred", ascending=False).head(k)
        scen[st] = {"products_ranked": int(len(prod)), "bestseller_top5_rec_rate": float(np.average(best.actual, weights=best.n)),
                    "model_top5_rec_rate": float(np.average(model.actual, weights=model.n)),
                    "all_products_rec_rate": float(np.average(prod.actual, weights=prod.n)),
                    "overlap": int(len(set(best.index) & set(model.index)))}
    out = {"n_train": len(train), "n_test": len(test), "test_period": [str(test.submission_time.min().date()), str(test.submission_time.max().date())],
           "results": res.round(4).to_dict("records"), "bootstrap_auc_diff_gb": boot,
           "permutation_importance_top15": imp.head(15).round(4).to_dict(), "scenario": scen}
    json.dump(out, open(ROOT / "reports" / "feature_value.json", "w"), indent=2)

    # ---------------- figures
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    r = res[res.feature_set != names[3]]
    x = np.arange(3); w = .36
    for j, (mname, col) in enumerate([("Logistic regression", GREY), (gb, ACCENT)]):
        v = r[r.model == mname].roc_auc.values
        ax.bar(x + (j - .5) * (w + .02), v, w, color=col, label=mname)
        for xi, vi in zip(x, v):
            ax.text(xi + (j - .5) * (w + .02), vi + .004, f"{vi:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [n.split(" (")[0].replace("A. ", "A. ").replace(" features", "") for n in names[:3]], fontsize=8.5)
    ax.set_ylim(.5, max(.8, r.roc_auc.max() + .04)); ax.set_ylabel("ROC-AUC (newest 20% of reviews)")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Each feature family improves out-of-time prediction")
    fig.tight_layout(); fig.savefig(FIG / "f13_feature_sets.png", dpi=200, facecolor="white"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    top = imp.head(12)[::-1]
    w4 = set(SETS[names[2]]["num"] + SETS[names[2]]["cat"])
    ax.barh([t.replace("_", " ") for t in top.index], top.values, color=[ACCENT if t in w4 else GREY for t in top.index], height=.6)
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled"); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ACCENT, label="Week 4 engineered"), Patch(color=GREY, label="Earlier features")],
              frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Most important features (gradient boosting, feature set C)")
    fig.tight_layout(); fig.savefig(FIG / "f14_importance.png", dpi=200, facecolor="white"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    sts = list(scen)
    x = np.arange(len(sts))
    for j, (key, lbl, col) in enumerate([("bestseller_top5_rec_rate", "Top 5 by popularity (loves)", GREY),
                                          ("model_top5_rec_rate", "Top 5 by skin-type-aware model", ACCENT)]):
        v = [scen[s][key] * 100 for s in sts]
        ax.bar(x + (j - .5) * (w + .02), v, w, color=col, label=lbl)
        for xi, vi in zip(x, v):
            ax.text(xi + (j - .5) * (w + .02), vi + .5, f"{vi:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, sts); ax.set_ylim(50, 112); ax.set_yticks(range(50, 101, 10)); ax.set_ylabel("Actual % recommended (test period)")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    ax.set_title("Scenario: what shoppers of each skin type would have experienced")
    fig.tight_layout(); fig.savefig(FIG / "f15_scenario.png", dpi=200, facecolor="white"); plt.close(fig)
    print(res.round(4).to_string()); print(json.dumps({"boot": boot, "scenario": scen}, indent=1)); print(imp.head(15).round(4))


if __name__ == "__main__":
    main()
