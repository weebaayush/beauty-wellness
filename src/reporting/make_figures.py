"""Create the Week 3 report figures from the pipeline outputs.  python -m src.reporting.make_figures"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "week3" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID, ACCENT, BEFORE = "#1a1a1a", "#6b6b6b", "#e6e6e6", "#E8590C", "#9a9a9a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titlelocation": "left"})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200, facecolor="white")
    plt.close(fig)


log = json.load(open(ROOT / "reports" / "cleaning_log.json"))
qa = json.load(open(ROOT / "reports" / "quality_before_after.json"))

# 1. rows by stage
sh = pd.DataFrame(log["shapes"])
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.barh(sh["stage"][::-1], sh["reviews"][::-1], color=ACCENT, height=0.55)
for y, v in enumerate(sh["reviews"][::-1]):
    ax.text(v + 400, y, f"{v:,}", va="center", color=INK, fontsize=9)
ax.set_xlim(0, sh["reviews"].max() * 1.15)
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_xlabel("Review rows")
ax.set_title("Review rows remaining after each pipeline stage")
save(fig, "fig2_rows_by_stage.png")

# 2. label consolidation (distinct values before vs after)
keys = [("brand_name_distinct_spellings", "Brand spellings"), ("category_distinct_labels", "Category labels"),
        ("skin_type_distinct_labels", "Skin-type labels"), ("skin_tone_distinct_labels", "Skin-tone labels")]
fig, ax = plt.subplots(figsize=(7.5, 3.0))
x = np.arange(len(keys)); w = 0.36
b = [qa["before"][k] for k, _ in keys]; a = [qa["after"][k] for k, _ in keys]
ax.bar(x - w / 2 - 0.01, b, w, color=BEFORE, label="Raw")
ax.bar(x + w / 2 + 0.01, a, w, color=ACCENT, label="Clean")
for i in range(len(keys)):
    ax.text(x[i] - w / 2, b[i] + 1, str(b[i]), ha="center", fontsize=9, color=INK)
    ax.text(x[i] + w / 2, a[i] + 1, str(a[i]), ha="center", fontsize=9, color=INK)
ax.set_xticks(x, [l for _, l in keys]); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_ylabel("Distinct values"); ax.legend(frameon=False)
ax.set_title("Inconsistent labels consolidated into canonical values")
save(fig, "fig3_label_consolidation.png")

# 3. skin type raw labels
raw = pd.concat(pd.read_csv(f, dtype=str, usecols=["skin_type"]) for f in glob.glob(str(ROOT / "data/raw/simulated/reviews_*.csv")))
vc = raw["skin_type"].fillna("(missing)").map(lambda s: repr(s) if s != "(missing)" else s).value_counts()
fig, ax = plt.subplots(figsize=(7.5, 3.6))
cols = [BEFORE if k == "(missing)" else ACCENT for k in vc.index]
ax.barh(vc.index[::-1], vc.values[::-1], color=cols[::-1], height=0.6)
for y, v in enumerate(vc.values[::-1]):
    ax.text(v + 150, y, f"{v:,}", va="center", fontsize=8, color=INK)
ax.set_xlim(0, vc.max() * 1.15); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_xlabel("Reviews"); ax.set_title("Raw skin_type values: 15 spellings for 4 real categories")
save(fig, "fig4_skin_type_raw.png")

# 4. missingness by column with chosen strategy
rev = pd.read_csv(ROOT / "data/processed/reviews_clean.csv.gz", low_memory=False)
prod = pd.read_csv(ROOT / "data/processed/products_clean.csv")
miss = pd.Series({
    "eye_color": rev["eye_color_missing"].mean(), "skin_type": rev["skin_type_missing"].mean(),
    "hair_color": rev["hair_color_missing"].mean(), "skin_tone": rev["skin_tone_missing"].mean(),
    "helpfulness (0 votes)": (rev["total_feedback_count"] == 0).mean(),
    "is_recommended": (~rev["target_available"]).mean(), "highlights (product)": prod["highlights_missing"].mean(),
    "size (product)": prod["size_missing"].mean(), "ingredients (product)": prod["ingredients_missing"].mean(),
    "price (product, raw)": prod["price_imputed"].mean(), "rating": rev["rating"].isna().mean()}).sort_values() * 100
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.barh(miss.index, miss.values, color=ACCENT, height=0.6)
for y, v in enumerate(miss.values):
    ax.text(v + 0.6, y, f"{v:.1f}%", va="center", fontsize=8, color=INK)
ax.set_xlim(0, miss.max() * 1.18); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_xlabel("% of rows missing"); ax.set_title("Missing values by field (each has its own strategy, Table 9)")
save(fig, "fig5_missingness.png")

# 5. price per ml by category, log scale, outliers highlighted
p = prod.dropna(subset=["price_per_ml"])
cats = p.groupby("secondary_category")["price_per_ml"].median().sort_values().index
fig, ax = plt.subplots(figsize=(7.5, 3.6))
rng = np.random.default_rng(0)
for i, c in enumerate(cats):
    d = p[p.secondary_category == c]
    jit = rng.uniform(-0.25, 0.25, len(d))
    norm = d[~d.flag_price_outlier]; out = d[d.flag_price_outlier]
    ax.scatter(norm.price_per_ml, i + jit[~d.flag_price_outlier.values], s=9, color=BEFORE, alpha=.55, linewidths=0)
    ax.scatter(out.price_per_ml, i + jit[d.flag_price_outlier.values], s=40, color=ACCENT, edgecolor="white", linewidths=1.5, zorder=3)
    ax.plot([d.price_per_ml.median()] * 2, [i - .32, i + .32], color=INK, lw=2)
ax.set_xscale("log"); ax.set_yticks(range(len(cats)), cats)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
ax.set_xlabel("Price per mL (USD, log scale)"); ax.xaxis.grid(True, color=GRID, which="both"); ax.set_axisbelow(True)
ax.scatter([], [], s=40, color=ACCENT, label="Flagged outlier (|robust z| > 3.5)"); ax.scatter([], [], s=9, color=BEFORE, label="Product")
ax.plot([], [], color=INK, lw=2, label="Category median"); ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Price-per-mL outliers are judged within each category")
save(fig, "fig6_price_outliers.png")

# 6. normalisation effect on loves_count
v = prod["loves_count"].dropna()
lg = np.log1p(v); rs = (lg - lg.median()) / (lg.quantile(.75) - lg.quantile(.25))
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
for ax, data, t in zip(axes, [v, lg, rs], ["Raw loves_count", "log1p(loves_count)", "log1p + RobustScaler"]):
    ax.hist(data, bins=40, color=ACCENT, edgecolor="white", linewidth=0.5)
    ax.set_title(t, fontsize=9); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel("Products")
fig.suptitle("Normalising a heavily skewed popularity variable", x=0.02, ha="left", fontweight="bold", fontsize=11, color=INK)
save(fig, "fig7_normalisation.png")
print("figures written to", FIG)
