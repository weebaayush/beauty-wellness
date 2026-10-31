"""Week 4 - exploratory data analysis: statistics + figures.

    python -m src.eda.run_eda

Reads data/processed/features.csv.gz, writes docs/week4/figures/*.png and reports/eda_results.json.
All tests use alpha = 0.05 / 6 = 0.0083 (Bonferroni over the six Week 1 hypotheses) and report effect sizes.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "week4" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
ALPHA = 0.05 / 6
INK, MUTED, GRID, ACCENT, GREY = "#1a1a1a", "#6b6b6b", "#e6e6e6", "#E8590C", "#9a9a9a"
SEQ = ["#FFF1E8", "#FDD0B1", "#F9A26C", "#E8590C", "#A33A06"]  # one-hue orange ramp
CAT = ["#E8590C", "#1a1a1a", "#2F80ED", "#9a9a9a"]  # fixed order: orange, ink, blue, grey
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 10.5, "axes.titlecolor": INK, "axes.titlelocation": "left"})
R: dict = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200, facecolor="white")
    plt.close(fig)


def grid(ax, axis="y"):
    for a in (["x", "y"] if axis == "both" else [axis]):
        getattr(ax, f"{a}axis").grid(True, color=GRID)
    ax.set_axisbelow(True)


def prop_ci(k, n, z=1.96):
    """Wilson score interval for a proportion."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


df = pd.read_csv(ROOT / "data/processed/features.csv.gz", low_memory=False, parse_dates=["submission_time"])
lab = df[df["target_available"]].copy()
lab["is_recommended"] = lab["is_recommended"].astype(int)
known = lab[lab["skin_type"] != "unknown"]
R["n_reviews"], R["n_labelled"], R["n_products"] = len(df), len(lab), df["product_id"].nunique()

# ------------------------------------------------------------ 1. univariate: targets
rc = df["rating"].value_counts().sort_index()
R["rating_share"] = (rc / rc.sum()).round(4).to_dict()
R["rating_mean"], R["rating_median"], R["rating_skew"] = float(df["rating"].mean()), float(df["rating"].median()), float(stats.skew(df["rating"].dropna()))
R["rec_rate"] = float(lab["is_recommended"].mean())
fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8), gridspec_kw={"width_ratios": [3, 1.3]})
ax = axes[0]
ax.bar(rc.index.astype(int).astype(str), rc.values / rc.sum() * 100, color=ACCENT, width=0.6)
for x, v in zip(rc.index.astype(int).astype(str), rc.values / rc.sum() * 100):
    ax.text(x, v + 1, f"{v:.1f}%", ha="center", fontsize=8, color=INK)
ax.set_xlabel("Star rating"); ax.set_ylabel("% of reviews"); grid(ax)
ax.set_title(f"Ratings are strongly positive (mean {R['rating_mean']:.2f}, skew {R['rating_skew']:.2f})")
ax = axes[1]
ax.bar(["Recommended", "Not"], [R["rec_rate"] * 100, (1 - R["rec_rate"]) * 100], color=[ACCENT, GREY], width=0.6)
for i, v in enumerate([R["rec_rate"] * 100, (1 - R["rec_rate"]) * 100]):
    ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8, color=INK)
ax.set_ylim(0, 100); grid(ax); ax.set_title("Target balance")
save(fig, "f01_targets.png")

# ------------------------------------------------------------ 2. review concentration (Lorenz)
cnt = df["product_id"].value_counts().sort_values()
cum = np.r_[0, cnt.cumsum().values / cnt.sum()]
xs = np.linspace(0, 1, len(cum))
gini = 1 - 2 * np.trapezoid(cum, xs)
top10 = cnt.sort_values(ascending=False)
R["gini_reviews"] = float(gini)
R["top_10pct_products_share"] = float(top10.iloc[: int(len(top10) * .1)].sum() / top10.sum())
R["top_20_products_share"] = float(top10.iloc[:20].sum() / top10.sum())
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.plot(xs * 100, cum * 100, color=ACCENT, lw=2)
ax.plot([0, 100], [0, 100], color=GREY, lw=1, ls="--")
ax.fill_between(xs * 100, cum * 100, xs * 100, color=ACCENT, alpha=.12)
ax.set_xlabel("% of products (fewest reviews first)"); ax.set_ylabel("% of reviews"); grid(ax, "both")
ax.text(5, 80, f"Gini = {gini:.2f}\nTop 10% of products\n= {R['top_10pct_products_share']*100:.0f}% of reviews", fontsize=8.5, color=INK)
ax.set_title("A few products dominate the reviews")
save(fig, "f02_concentration.png")

# ------------------------------------------------------------ 3. H1 heatmap category x skin type + test
cats = lab.groupby("secondary_category").size().sort_values(ascending=False).index
types = ["dry", "normal", "combination", "oily"]
piv = known.pivot_table(index="secondary_category", columns="skin_type", values="is_recommended", aggfunc="mean").loc[cats, types] * 100
npiv = known.pivot_table(index="secondary_category", columns="skin_type", values="is_recommended", aggfunc="size").loc[cats, types]
fig, ax = plt.subplots(figsize=(6.2, 3.4))
im = ax.imshow(piv.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("o", SEQ), aspect="auto",
               vmin=np.nanmin(piv.values), vmax=np.nanmax(piv.values))
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(j, i, f"{v:.0f}%\nn={npiv.values[i, j]:,}", ha="center", va="center", fontsize=7.5,
                color="white" if v > np.nanpercentile(piv.values, 70) else INK)
ax.set_xticks(range(len(types)), types); ax.set_yticks(range(len(cats)), cats)
ax.set_title("Recommendation rate by category × skin type (H1)")
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=.04); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
save(fig, "f03_h1_heatmap.png")
groups = [known.loc[known.skin_type == t, "rating"].dropna() for t in types]
H, p = stats.kruskal(*groups)
n = sum(len(g) for g in groups)
eps2 = (H - len(groups) + 1) / (n - len(groups))
pairs = []
for a, b in combinations(types, 2):
    u, pp = stats.mannwhitneyu(known.loc[known.skin_type == a, "rating"].dropna(), known.loc[known.skin_type == b, "rating"].dropna())
    pairs.append({"pair": f"{a} vs {b}", "p_bonf": float(min(1, pp * 6))})
R["H1"] = {"kruskal_H": float(H), "p": float(p), "epsilon_sq": float(eps2), "n": int(n),
           "mean_rating": {t: float(g.mean()) for t, g in zip(types, groups)}, "pairwise": pairs,
           "rec_rate": (known.groupby("skin_type")["is_recommended"].mean()).round(4).to_dict()}

# ------------------------------------------------------------ 4. H2 price vs satisfaction (product level)
prod = lab.groupby("product_id").agg(price_per_ml=("price_per_ml", "first"), rating=("rating", "mean"), n=("rating", "size"),
                                     cat=("secondary_category", "first"), rec=("is_recommended", "mean")).dropna()
prod = prod[prod.n >= 10]
rho, prho = stats.spearmanr(prod.price_per_ml, prod.rating)
z = np.arctanh(rho); se = 1 / np.sqrt(len(prod) - 3)
R["H2"] = {"spearman_rho": float(rho), "p": float(prho), "ci95": [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))], "n_products": int(len(prod))}
# within-category rho
R["H2"]["by_category"] = {c: float(stats.spearmanr(g.price_per_ml, g.rating)[0]) for c, g in prod.groupby("cat") if len(g) >= 15}
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.scatter(prod.price_per_ml, prod.rating, s=np.clip(prod.n / 6, 6, 120), color=ACCENT, alpha=.45, edgecolor="white", linewidths=.6)
lx = np.log(prod.price_per_ml)
coef = np.polyfit(lx, prod.rating, 1)
xx = np.linspace(lx.min(), lx.max(), 50)
ax.plot(np.exp(xx), np.polyval(coef, xx), color=INK, lw=2)
ax.set_xscale("log"); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"${v:g}"))
ax.set_xlabel("Price per mL (log scale)"); ax.set_ylabel("Product mean rating"); grid(ax, "both")
ax.set_title(f"Price barely relates to satisfaction (H2): Spearman ρ = {rho:.2f}, 95% CI [{R['H2']['ci95'][0]:.2f}, {R['H2']['ci95'][1]:.2f}]")
ax.text(.99, .04, "Bubble size = number of reviews; products with ≥ 10 reviews", transform=ax.transAxes, ha="right", fontsize=7.5, color=MUTED)
save(fig, "f04_h2_price.png")

# ------------------------------------------------------------ 5. H4 fragrance x skin type (+ LR interaction test)
sub = known[known.has_fragrance.notna()].copy()
fig, ax = plt.subplots(figsize=(7.5, 3.0))
w = 0.38
res4 = {}
for j, (val, lbl, col) in enumerate([(0, "Fragrance-free formula", GREY), (1, "Contains fragrance", ACCENT)]):
    rates, lo, hi = [], [], []
    for t in types:
        g = sub[(sub.skin_type == t) & (sub.has_fragrance == val)]["is_recommended"]
        r = g.mean(); a, b = prop_ci(g.sum(), len(g))
        rates.append(r * 100); lo.append((r - a) * 100); hi.append((b - r) * 100)
        res4[f"{t}_{'frag' if val else 'free'}"] = {"rate": float(r), "n": int(len(g))}
    x = np.arange(len(types)) + (j - .5) * (w + .02)
    ax.bar(x, rates, w, color=col, label=lbl, yerr=[lo, hi], error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
ax.set_xticks(range(len(types)), types); ax.set_ylim(60, 95); ax.set_ylabel("% recommended"); grid(ax)
ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
ax.set_title("Fragrance lowers recommendation only for dry skin (H4); bars show 95% CI")
save(fig, "f05_h4_fragrance.png")
dry = sub[sub.skin_type == "dry"]
tab = pd.crosstab(dry.has_fragrance, dry.is_recommended)
chi2, p4, _, _ = stats.chi2_contingency(tab)
V = np.sqrt(chi2 / tab.values.sum())
# likelihood-ratio test of skin_type x fragrance interaction
X0 = pd.get_dummies(sub[["skin_type"]], drop_first=True).astype(float).assign(frag=sub.has_fragrance.values)
X1 = X0.copy()
for c in [c for c in X0.columns if c.startswith("skin_type_")]:
    X1[c + "_x_frag"] = X0[c] * X0["frag"]
yv = sub.is_recommended.values


def loglik(X):
    m = LogisticRegression(penalty=None, max_iter=2000).fit(X, yv)
    pr = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return np.sum(yv * np.log(pr) + (1 - yv) * np.log(1 - pr))


LR = 2 * (loglik(X1) - loglik(X0))
p_int = stats.chi2.sf(LR, df=X1.shape[1] - X0.shape[1])
R["H4"] = {"rates": res4, "dry_chi2": float(chi2), "dry_p": float(p4), "dry_cramers_v": float(V),
           "dry_gap_pp": float((res4["dry_free"]["rate"] - res4["dry_frag"]["rate"]) * 100),
           "interaction_LR": float(LR), "interaction_p": float(p_int)}

# ------------------------------------------------------------ 6. skin match / mismatch
mm = lab.assign(fit=np.select([lab.skin_mismatch == 1, lab.skin_match == 1, lab.skin_match == 0],
                              ["Opposite skin type", "Matches claim", "Other / neutral"], "No claim or unknown skin"))
order = ["Matches claim", "Other / neutral", "No claim or unknown skin", "Opposite skin type"]
agg = mm.groupby("fit")["is_recommended"].agg(["sum", "count"]).loc[order]
fig, ax = plt.subplots(figsize=(7.5, 2.6))
r = agg["sum"] / agg["count"] * 100
ci = [prop_ci(s, n) for s, n in zip(agg["sum"], agg["count"])]
ax.barh(order[::-1], r.values[::-1], color=[GREY, GREY, GREY, ACCENT][::-1] if False else [ACCENT, GREY, GREY, INK][::-1], height=.55,
        xerr=[[(r.values[i] - ci[i][0] * 100) for i in range(4)][::-1], [(ci[i][1] * 100 - r.values[i]) for i in range(4)][::-1]],
        error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
for y, (v, nn) in enumerate(zip(r.values[::-1], agg["count"].values[::-1])):
    ax.text(v + 4, y, f"{v:.1f}%  (n={nn:,})", va="center", fontsize=8, color=INK)
ax.set_xlim(0, 118); ax.set_xlabel("% recommended"); grid(ax, "x")
ax.set_title("Skin-type fit is one of the strongest signals")
save(fig, "f06_skin_fit.png")
R["skin_fit"] = {k: {"rate": float(s / n), "n": int(n)} for k, s, n in zip(order, agg["sum"], agg["count"])}

# ------------------------------------------------------------ 7. H6 helpfulness
voted = df[df.total_feedback_count > 0].copy()
q75 = voted.helpfulness_smoothed.quantile(.75)
voted["top_helpful"] = voted.helpfulness_smoothed >= q75
a, b = voted[voted.top_helpful], voted[~voted.top_helpful]
u, p6 = stats.mannwhitneyu(a.review_word_count, b.review_word_count)
rbc = 1 - 2 * u / (len(a) * len(b))
neg_a, neg_b = (a.rating <= 2).mean(), (b.rating <= 2).mean()
chi6, p6b, _, _ = stats.chi2_contingency(pd.crosstab(voted.top_helpful, voted.rating <= 2))
R["H6"] = {"n_voted": int(len(voted)), "median_words_top": float(a.review_word_count.median()), "median_words_other": float(b.review_word_count.median()),
           "p_length": float(p6), "rank_biserial": float(rbc), "neg_share_top": float(neg_a), "neg_share_other": float(neg_b), "p_negative": float(p6b)}
fig, ax = plt.subplots(figsize=(7.5, 2.6))
share = voted.groupby("rating")["top_helpful"].mean() * 100
ax.bar(share.index.astype(int).astype(str), share.values, color=[ACCENT if i <= 2 else GREY for i in share.index.astype(int)], width=.6)
for x, v in zip(share.index.astype(int).astype(str), share.values):
    ax.text(x, v + .8, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
ax.set_xlabel("Star rating (reviews with ≥ 1 vote)"); ax.set_ylabel("% in top helpfulness quartile"); grid(ax)
ax.set_title("Critical reviews are more often voted helpful (H6)")
save(fig, "f07_h6_helpful.png")

# ------------------------------------------------------------ 8. time trends (small multiples, no dual axis)
mon = df.set_index("submission_time").resample("QE").agg(reviews=("rating", "size"), rating=("rating", "mean"))
fig, axes = plt.subplots(2, 1, figsize=(7.5, 3.6), sharex=True)
axes[0].bar(mon.index, mon.reviews, width=70, color=ACCENT); axes[0].set_ylabel("Reviews"); grid(axes[0])
axes[0].set_title("Quarterly review volume and mean rating")
axes[1].plot(mon.index, mon.rating, color=INK, lw=2, marker="o", ms=3); axes[1].set_ylabel("Mean rating"); grid(axes[1])
axes[1].set_ylim(3.8, 4.3)
save(fig, "f08_time.png")
R["time"] = {"quarterly_rating_min": float(mon.rating.min()), "quarterly_rating_max": float(mon.rating.max())}
mon_m = df.groupby(df.submission_time.dt.month)["rating"].mean()
R["seasonality_month_range"] = float(mon_m.max() - mon_m.min())

# ------------------------------------------------------------ 9. aspects by star rating
aspects = [c for c in df.columns if c.startswith("aspect_")]
asp = df.groupby("rating")[aspects].apply(lambda g: (g > 0).mean() * 100)
asp.columns = [c.replace("aspect_", "").replace("_", " ") for c in asp.columns]
fig, ax = plt.subplots(figsize=(7.0, 2.8))
im = ax.imshow(asp.T.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("o", SEQ), aspect="auto")
for i in range(asp.shape[1]):
    for j in range(asp.shape[0]):
        v = asp.T.values[i, j]
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=7.5, color="white" if v > 45 else INK)
ax.set_xticks(range(asp.shape[0]), [f"{int(r)}★" for r in asp.index]); ax.set_yticks(range(asp.shape[1]), asp.columns)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Share of reviews mentioning each aspect, by star rating")
save(fig, "f09_aspects.png")
R["aspects_by_rating"] = asp.round(1).to_dict()

# ------------------------------------------------------------ 10. correlation heatmap of engineered features
num = ["price_rel_category", "n_ingredients", "strong_actives", "product_prior_bayes_rating", "product_momentum", "review_velocity",
       "brand_te", "profile_completeness", "review_word_count", "helpfulness_smoothed", "aspect_skin_reaction_neg",
       "skin_match", "dry_x_fragrance", "skin_mismatch", "is_recommended"]
cm = lab[num].corr(method="spearman")
fig, ax = plt.subplots(figsize=(7.2, 5.6))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("d", ["#2F80ED", "#F2F2F2", "#E8590C"])
im = ax.imshow(np.where(np.tril(np.ones_like(cm.values), 0) > 0, cm.values, np.nan), cmap=cmap, vmin=-1, vmax=1)
for i in range(len(num)):
    for j in range(len(num)):
        if i > j:
            ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color=INK)
lbls = [s.replace("_", " ") for s in num]
ax.set_xticks(range(len(num)), lbls, rotation=60, ha="right", fontsize=7); ax.set_yticks(range(len(num)), lbls, fontsize=7)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=.035); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
ax.set_title("Spearman correlations (blue = negative, orange = positive)")
save(fig, "f10_corr.png")
R["corr_with_target"] = cm["is_recommended"].drop("is_recommended").round(3).to_dict()
hi = [(a, b, round(float(cm.loc[a, b]), 2)) for a, b in combinations(num[:-1], 2) if abs(cm.loc[a, b]) > .6]
R["high_collinearity_pairs"] = hi

# ------------------------------------------------------------ 11. mutual information
mi_cols = [c for c in num if c != "is_recommended"] + ["n_actives_present", "price_per_ml", "loves_count", "is_holiday_season",
                                                       "is_early_adopter", "fragrance_free_verified", "premium_x_first_purchase"]
Xmi = lab[mi_cols].fillna(lab[mi_cols].median())
mi = pd.Series(mutual_info_classif(Xmi, lab.is_recommended, random_state=42, n_neighbors=5), index=mi_cols).sort_values()
R["mutual_info"] = mi.round(4).to_dict()
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.barh([m.replace("_", " ") for m in mi.index], mi.values, color=[ACCENT if v >= mi.quantile(.7) else GREY for v in mi.values], height=.6)
ax.set_xlabel("Mutual information with is_recommended (nats)"); grid(ax, "x")
ax.set_title("Which features carry information about recommendation?")
save(fig, "f11_mutual_info.png")

# ------------------------------------------------------------ 12. product segments (PCA + k-means)
pf = lab.groupby("product_id").agg(price_rel=("price_rel_category", "first"), actives=("strong_actives", "first"),
                                   n_ing=("n_ingredients", "first"), frag=("has_fragrance", "first"),
                                   ff=("claim_fragrance_free", "first"), vegan=("claim_vegan", "first"),
                                   rec=("is_recommended", "mean"), reviews=("rating", "size"), cat=("secondary_category", "first")).dropna()
pf = pf[pf.reviews >= 10]
feats = ["price_rel", "actives", "n_ing", "frag", "ff", "vegan"]
Z = StandardScaler().fit_transform(np.c_[np.log(pf.price_rel), pf[feats[1:]].astype(float)])
inertia = {k: float(KMeans(k, n_init=10, random_state=42).fit(Z).inertia_) for k in range(2, 8)}
k = 4
km = KMeans(k, n_init=10, random_state=42).fit(Z)
pf["segment"] = km.labels_
pc = PCA(2, random_state=42).fit(Z)
P = pc.transform(Z)
prof = pf.groupby("segment").agg(products=("rec", "size"), price_rel=("price_rel", "median"), strong_actives=("actives", "mean"),
                                 fragranced=("frag", "mean"), fragrance_free_claim=("ff", "mean"), vegan=("vegan", "mean"), rec_rate=("rec", "mean"))
R["segments"] = {"k": k, "inertia": inertia, "pca_var": pc.explained_variance_ratio_.round(3).tolist(), "profile": prof.round(3).reset_index().to_dict("records")}
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for s in range(k):
    m = pf.segment.values == s
    ax.scatter(P[m, 0], P[m, 1], s=14, color=CAT[s], alpha=.7, label=f"Segment {s + 1} (n={m.sum()})", edgecolor="white", linewidths=.4)
ax.set_xlabel(f"PC1 ({pc.explained_variance_ratio_[0]*100:.0f}% of variance)"); ax.set_ylabel(f"PC2 ({pc.explained_variance_ratio_[1]*100:.0f}%)")
grid(ax, "both"); ax.legend(frameon=False, fontsize=7.5, loc="upper right")
ax.set_title("Product segments (k-means on formulation & price, shown on 2 PCs)")
save(fig, "f12_segments.png")

json.dump(R, open(ROOT / "reports" / "eda_results.json", "w"), indent=2, default=str)
print(json.dumps({k: R[k] for k in ["H1", "H2", "H4", "H6", "skin_fit", "gini_reviews", "top_10pct_products_share"]}, indent=1, default=str))
