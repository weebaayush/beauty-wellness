"""Week 2 - first data-quality profile of the Sephora dataset (S1).

Usage:
    python src/data_quality.py
Reads data/raw/sephora/*.csv and writes data/quality_report.md plus
data/interim/sephora_reviews.parquet (merged, typed review-level table).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "sephora"
OUT = ROOT / "data"


def load():
    products = pd.read_csv(SRC / "product_info.csv")
    parts = sorted(SRC.glob("reviews_*.csv"))
    if not parts:
        raise FileNotFoundError("Run `python src/download_data.py sephora` first.")
    reviews = pd.concat((pd.read_csv(p, index_col=0, low_memory=False) for p in parts), ignore_index=True)
    return products, reviews


def merge(products, reviews):
    keep = ["product_id", "ingredients", "highlights", "primary_category", "secondary_category",
            "tertiary_category", "loves_count", "size", "sale_price_usd", "limited_edition",
            "new", "online_only", "sephora_exclusive"]
    keep = [c for c in keep if c in products.columns]
    df = reviews.merge(products[keep], on="product_id", how="left")
    if "submission_time" in df:
        df["submission_time"] = pd.to_datetime(df["submission_time"], errors="coerce")
    for c in ["skin_type", "skin_tone", "eye_color", "hair_color", "brand_name", "secondary_category"]:
        if c in df:
            df[c] = df[c].astype("category")
    return df


def checks(products, df):
    lines = ["# Data quality report - Sephora (S1)", ""]
    lines += [f"- Products: **{len(products):,}**", f"- Reviews (merged): **{len(df):,}**",
              f"- Columns: {df.shape[1]}", ""]

    lines += ["## Completeness (% missing)", "", "| Column | % missing |", "|---|---|"]
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    lines += [f"| {c} | {v:.1f} |" for c, v in miss.items()]

    dup_cols = [c for c in ["author_id", "product_id", "review_text"] if c in df]
    lines += ["", "## Uniqueness", "",
              f"- Exact duplicate rows: {df.duplicated().sum():,}",
              f"- Duplicates on {dup_cols}: {df.duplicated(subset=dup_cols).sum():,}"]

    lines += ["", "## Validity", ""]
    if "rating" in df:
        lines.append(f"- Ratings outside 1-5: {(~df['rating'].between(1, 5)).sum():,}")
    if "price_usd" in df:
        lines.append(f"- Non-positive prices: {(df['price_usd'] <= 0).sum():,}")
    for c in ["skin_type", "skin_tone"]:
        if c in df:
            lines.append(f"- {c} levels: {', '.join(map(str, df[c].cat.categories))}")

    lines += ["", "## Consistency", ""]
    if {"rating", "is_recommended"} <= set(df.columns):
        ct = pd.crosstab(df["rating"], df["is_recommended"], dropna=False)
        try:
            table = ct.to_markdown()
        except ImportError:  # tabulate not installed
            table = "```\n" + ct.to_string() + "\n```"
        lines += ["Rating x is_recommended:", "", table, ""]
        odd = ((df["rating"] <= 2) & (df["is_recommended"] == 1)).sum()
        lines.append(f"- 1-2 star reviews that still recommend: {odd:,}")

    lines += ["", "## Timeliness", ""]
    if "submission_time" in df:
        yr = df["submission_time"].dt.year.value_counts().sort_index()
        lines += ["| Year | Reviews |", "|---|---|"] + [f"| {int(y)} | {n:,} |" for y, n in yr.items()]

    lines += ["", "## Targets", ""]
    if "rating" in df:
        lines.append("Rating share (%): " + ", ".join(
            f"{int(k)}*: {v:.1f}" for k, v in (df["rating"].value_counts(normalize=True).sort_index() * 100).items()))
    if "is_recommended" in df:
        lines.append(f"\nRecommendation rate: {df['is_recommended'].mean() * 100:.1f}%")
    return "\n".join(lines)


if __name__ == "__main__":
    products, reviews = load()
    df = merge(products, reviews)
    (OUT / "interim").mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(OUT / "interim" / "sephora_reviews.parquet", index=False)
    except ImportError:  # pyarrow not installed
        df.to_csv(OUT / "interim" / "sephora_reviews.csv.gz", index=False)
    report = checks(products, df)
    (OUT / "quality_report.md").write_text(report, encoding="utf-8")
    print(report[:2000])
