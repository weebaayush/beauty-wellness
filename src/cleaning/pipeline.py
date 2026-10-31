"""Run the full cleaning & preprocessing pipeline.

    python -m src.cleaning.pipeline                     # uses configs/cleaning.yaml
    python -m src.cleaning.pipeline --raw data/raw/sephora

Outputs
    data/processed/reviews_clean.csv(.gz)   review-level analysis table
    data/processed/products_clean.csv       product-level table
    data/processed/X_train.csv, X_test.csv  scaled & encoded model matrices (+ y_*.csv)
    reports/cleaning_log.json / .md         every action with rows affected
    reports/data_quality_before_after.md    quality metrics before vs after
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml

from . import steps
from .impute_outliers import engineer_features, handle_missing, treat_outliers
from .log import CleaningLog
from .preprocess import prepare_model_matrix

ROOT = Path(__file__).resolve().parents[2]


def load_raw(raw_dir: Path):
    prod = pd.read_csv(raw_dir / "product_info.csv")
    parts = sorted(raw_dir.glob("reviews_*.csv"))
    rev = pd.concat((pd.read_csv(p, low_memory=False, dtype={"author_id": str}) for p in parts), ignore_index=True)
    return prod, rev


def quality_snapshot(prod: pd.DataFrame, rev: pd.DataFrame) -> dict:
    def pct(s):
        return round(100 * s, 2)
    snap = {
        "products": len(prod), "reviews": len(rev),
        "duplicate_product_ids": int(prod["product_id"].duplicated().sum()),
        "product_price_missing_or_invalid_%": pct(pd.to_numeric(prod["price_usd"], errors="coerce").pipe(lambda p: p.isna() | (p < 1) | (p > 1000)).mean()),
        "brand_name_distinct_spellings": int(prod["brand_name"].nunique()),
        "category_distinct_labels": int(prod["secondary_category"].nunique()),
        "skin_type_distinct_labels": int(rev["skin_type"].dropna().nunique()),
        "skin_tone_distinct_labels": int(rev["skin_tone"].dropna().nunique()),
        "skin_type_missing_%": pct(rev["skin_type"].isna().mean()),
        "is_recommended_missing_%": pct(rev["is_recommended"].isna().mean()),
        "rating_invalid_(text_or_out_of_range)": int((rev["rating"].notna() & ~pd.to_numeric(rev["rating"], errors="coerce").between(1, 5)).sum()),
        "rating_missing": int(rev["rating"].isna().sum()),
        "review_text_with_html_%": pct(rev["review_text"].fillna("").str.contains(r"<[^>]+>", regex=True).mean()),
        "review_text_with_email_pii": int(rev["review_text"].fillna("").str.contains(r"@\w+\.", regex=True).sum()),
        "orphan_reviews": int((~rev["product_id"].isin(prod["product_id"])).sum()),
    }
    return snap


def run(cfg_path: Path, raw_override: str | None = None, sample: int | None = None):
    cfg = yaml.safe_load(cfg_path.read_text())
    raw_dir = ROOT / (raw_override or cfg["paths"]["raw_dir"])
    proc = ROOT / cfg["paths"]["processed_dir"]; proc.mkdir(parents=True, exist_ok=True)
    rep = ROOT / cfg["paths"]["reports_dir"]; rep.mkdir(parents=True, exist_ok=True)
    log = CleaningLog()

    prod, rev = load_raw(raw_dir)
    if sample:
        rev = rev.sample(sample, random_state=cfg["seed"])
    before = quality_snapshot(prod, rev)
    log.shape("1. raw", len(prod), len(rev))

    steps.check_schema(prod, steps.REQUIRED_PRODUCT_COLS, "products", log)
    steps.check_schema(rev, steps.REQUIRED_REVIEW_COLS, "reviews", log)

    prod = steps.standardise_products(prod, cfg, log)
    rev = steps.standardise_reviews(rev, cfg, log)
    log.shape("2. standardised", len(prod), len(rev))

    prod, rev = steps.apply_validity_rules(prod, rev, cfg, log)
    log.shape("3. validity rules", len(prod), len(rev))

    prod, rev = steps.remove_duplicates(prod, rev, log)
    rev = steps.enforce_integrity(prod, rev, log)
    log.shape("4. de-duplicated & integrity", len(prod), len(rev))

    rev = steps.clean_review_text(rev, cfg, log)
    prod = steps.clean_ingredients(prod, cfg, log)
    log.shape("5. text & ingredients cleaned", len(prod), len(rev))

    prod, rev = handle_missing(prod, rev, cfg, log)
    log.shape("6. missing values handled", len(prod), len(rev))

    prod, rev = treat_outliers(prod, rev, cfg, log)
    log.shape("7. outliers treated", len(prod), len(rev))

    df = engineer_features(prod, rev, cfg, log)
    log.shape("8. features engineered (merged)", len(prod), len(df))

    # final validation - the pipeline fails loudly rather than silently producing bad data
    assert df["rating"].dropna().between(1, 5).all(), "rating out of range after cleaning"
    assert df["product_id"].isin(prod["product_id"]).all(), "orphan reviews remain"
    assert not df.duplicated(subset=["author_hash", "product_id", "review_text_clean"]).any(), "duplicates remain"
    assert not df["review_text_clean"].str.contains(r"[\w.+-]+@[\w-]+\.[\w.]+", regex=True).any(), "e-mail PII remains"
    log.add("validation", "final assertions passed (ranges, integrity, duplicates, PII)", len(df))

    X_tr, y_tr, X_te, y_te, pre = prepare_model_matrix(df)
    log.add("normalise", f"model matrix: {X_tr.shape[1]} features; train {len(X_tr):,} / test {len(X_te):,} (time-based split)", len(X_tr) + len(X_te))
    log.shape("9. model-ready (train + test)", len(prod), len(X_tr) + len(X_te))

    # outputs
    prod_out = prod.drop(columns=["ingredient_list", "highlights_list"]).assign(
        ingredient_list=prod["ingredient_list"].str.join("; "), highlights_list=prod["highlights_list"].str.join("; "))
    prod_out.to_csv(proc / "products_clean.csv", index=False)
    df.to_csv(proc / "reviews_clean.csv.gz", index=False, compression="gzip")
    for name, obj in [("X_train", X_tr), ("X_test", X_te), ("y_train", y_tr), ("y_test", y_te)]:
        obj.to_csv(proc / f"{name}.csv.gz", index=False, compression="gzip")
    with open(proc / "preprocessor.pkl", "wb") as f:
        pickle.dump(pre, f)

    after = quality_snapshot(prod.assign(price_usd=prod["price_usd"]),
                             df.rename(columns={"review_text_clean": "_t"}).assign(
                                 review_text=df["review_text_clean"],
                                 skin_type=df["skin_type"].where(df["skin_type"] != "unknown"),
                                 skin_tone=df["skin_tone"].where(df["skin_tone"] != "unknown"),
                                 is_recommended=df["is_recommended"].astype(float)))
    meta = {"config": str(cfg_path.relative_to(ROOT)), "raw_dir": str(raw_dir.relative_to(ROOT)),
            "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:16],
            "pandas": pd.__version__}
    log.to_json(rep / "cleaning_log.json", meta)
    (rep / "cleaning_log.md").write_text("# Cleaning log\n\n" + log.to_markdown(), encoding="utf-8")
    rows = [f"| {k} | {before[k]} | {after[k]} |" for k in before]
    (rep / "data_quality_before_after.md").write_text(
        "# Data quality: before vs after\n\n| Metric | Raw | Clean |\n|---|---|---|\n" + "\n".join(rows), encoding="utf-8")
    json.dump({"before": before, "after": after}, open(rep / "quality_before_after.json", "w"), indent=2)
    return df, prod, log, before, after


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cleaning.yaml")
    ap.add_argument("--raw", default=None, help="override raw data folder")
    ap.add_argument("--sample", type=int, default=None, help="use a random sample of N reviews (quick runs)")
    a = ap.parse_args()
    df, prod, log, before, after = run(ROOT / a.config, a.raw, a.sample)
    print(log.to_markdown())
    print(json.dumps({"before": before, "after": after}, indent=2))
