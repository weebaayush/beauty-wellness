"""Missing-value handling, outlier treatment and feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .log import CleaningLog


# ---------------------------------------------------------------- missing values

def handle_missing(prod: pd.DataFrame, rev: pd.DataFrame, cfg: dict, log: CleaningLog):
    unk = cfg["missing"]["unknown_label"]
    prod, rev = prod.copy(), rev.copy()

    # 1) Price (product): recover from the review table first (reviews carry a price snapshot),
    #    then fall back to the median of the same brand + category, then category. Flag every imputation.
    prod["price_imputed"] = prod["price_usd"].isna()
    from_reviews = rev.groupby("product_id")["price_usd"].median()
    cand = prod["product_id"].map(from_reviews)
    cand = cand.where(cand.between(cfg["validity"]["price_min_usd"], cfg["validity"]["price_max_usd"]))
    fill1 = prod["price_usd"].isna() & cand.notna()
    prod.loc[fill1, "price_usd"] = cand[fill1]
    log.add("missing", "missing product price recovered from review-level price snapshot", int(fill1.sum()))
    k = cfg["missing"]["min_group_size_for_impute"]
    for keys, label in [(["brand_name", "secondary_category"], "brand x category median"),
                        (["secondary_category"], "category median")]:
        grp = prod.groupby(keys)["price_usd"]
        med = grp.transform("median").where(grp.transform("count") >= k)
        fill = prod["price_usd"].isna() & med.notna()
        prod.loc[fill, "price_usd"] = med[fill]
        log.add("missing", f"remaining missing price imputed with {label} (groups >= {k})", int(fill.sum()))

    # 2) Size: cannot be inferred reliably -> left missing, flagged; price_per_ml will be missing too
    prod["size_missing"] = prod["size_ml"].isna()
    log.add("missing", "size left missing and flagged (not imputable without guessing)", int(prod["size_missing"].sum()))

    # 3) Ingredients / highlights: missing != 'none'; keep a missing indicator so models can learn from it
    prod["ingredients_missing"] = prod["ingredient_list"].str.len() == 0
    prod["highlights_missing"] = prod["highlights"].isna()
    log.add("missing", "ingredients missing -> indicator column added", int(prod["ingredients_missing"].sum()))
    log.add("missing", "highlights missing -> indicator column added", int(prod["highlights_missing"].sum()))

    # 4) Reviewer attributes: optional fields (likely MNAR) -> explicit 'unknown' category + indicator
    for col in ["skin_type", "skin_tone", "eye_color", "hair_color"]:
        miss = rev[col].isna() | (rev[col] == unk)
        rev[f"{col}_missing"] = miss
        rev[col] = rev[col].fillna(unk)
        log.add("missing", f"{col} missing -> '{unk}' category + indicator", int(miss.sum()))

    # 5) Helpfulness is missing *by design* when nobody voted: replace with a smoothed score
    #    (Laplace / Bayesian): (pos + 1) / (total + 2). 0 votes -> 0.5 (neutral), 1/1 -> 0.67, 30/30 -> 0.97
    n0 = rev["helpfulness"].isna().sum()
    rev["helpfulness_smoothed"] = (rev["total_pos_feedback_count"] + 1) / (rev["total_feedback_count"] + 2)
    log.add("missing", "helpfulness missing (0 votes) -> Laplace-smoothed helpfulness score", int(n0))

    # 6) Targets: never imputed (imputing the label would invent data). Rows kept for EDA, excluded from modelling.
    rev["target_available"] = rev["is_recommended"].notna()
    log.add("missing", "is_recommended missing -> kept for EDA, excluded from model training", int((~rev["target_available"]).sum()))
    rating_missing = rev["rating"].isna()
    log.add("missing", "rating missing (after validity checks) -> kept, excluded from rating analyses", int(rating_missing.sum()))
    return prod, rev


# ---------------------------------------------------------------- outliers

def robust_z(x: pd.Series) -> pd.Series:
    """Modified z-score (Iglewicz & Hoaglin): 0.6745 * (x - median) / MAD."""
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(0.0, index=x.index)
    return 0.6745 * (x - med) / mad


def treat_outliers(prod: pd.DataFrame, rev: pd.DataFrame, cfg: dict, log: CleaningLog):
    o = cfg["outliers"]
    prod, rev = prod.copy(), rev.copy()
    # Price per ml: compare within category on the log scale (prices are right-skewed)
    prod["price_per_ml"] = prod["price_usd"] / prod["size_ml"]
    logp = np.log(prod["price_per_ml"])
    prod["price_per_ml_rz"] = logp.groupby(prod["secondary_category"]).transform(robust_z)
    prod["flag_price_outlier"] = prod["price_per_ml_rz"].abs() > o["robust_z_threshold"]
    log.add("outliers", f"price-per-ml outliers flagged (|robust z| > {o['robust_z_threshold']} within category, log scale)",
            int(prod["flag_price_outlier"].sum()))
    # IQR rule on loves_count (log) - reported for comparison
    l = np.log1p(prod["loves_count"])
    q1, q3 = l.quantile([.25, .75])
    iqr_flag = (l < q1 - 1.5 * (q3 - q1)) | (l > q3 + 1.5 * (q3 - q1))
    log.add("outliers", "loves_count outliers by IQR rule on log scale (flagged, kept: viral products are real)", int(iqr_flag.sum()))
    prod["flag_loves_outlier"] = iqr_flag
    # Winsorise for modelling copies only (raw values kept for EDA)
    for col in ["price_per_ml", "loves_count"]:
        lo, hi = prod[col].quantile([o["winsor_lower_pct"], o["winsor_upper_pct"]])
        clipped = prod[col].clip(lo, hi)
        log.add("outliers", f"{col} winsorised at {int(o['winsor_lower_pct']*100)}th/{int(o['winsor_upper_pct']*100)}th percentiles (model copy)",
                int((clipped != prod[col]).sum() - prod[col].isna().sum()))
        prod[f"{col}_w"] = clipped
    # Review length: very long reviews are real; cap for modelling only
    hi = rev["review_word_count"].quantile(o["winsor_upper_pct"])
    rev["review_word_count_w"] = rev["review_word_count"].clip(upper=hi)
    log.add("outliers", "review word count capped at 99th percentile (model copy)", int((rev["review_word_count"] > hi).sum()))
    return prod, rev


# ---------------------------------------------------------------- feature engineering

def engineer_features(prod: pd.DataFrame, rev: pd.DataFrame, cfg: dict, log: CleaningLog) -> pd.DataFrame:
    f = cfg["features"]
    prod = prod.copy()
    for ing in f["key_actives"]:
        prod[f"has_{ing.replace(' ', '_').replace('.', '')}"] = prod["ingredient_list"].apply(lambda l, i=ing: i in l)
    prod["n_key_actives"] = prod["ingredient_list"].apply(lambda l: sum(i in l for i in f["key_actives"]))
    prod["has_fragrance"] = prod["ingredient_list"].apply(lambda l: float(any(i in l for i in f["potential_irritants"][:4])))
    prod.loc[prod["ingredients_missing"], "has_fragrance"] = np.nan   # unknown, not "fragrance-free"
    # claim flags from highlights
    for claim in ["Fragrance Free", "Vegan", "Clean at Sephora", "Hydrating", "Cruelty-Free", "Oil Free"]:
        prod[f"claim_{claim.lower().replace(' ', '_').replace('-', '_')}"] = prod["highlights_list"].apply(lambda h, c=claim: c in h)
    prod["good_for"] = prod["highlights_list"].apply(
        lambda h: next((x.split(":")[1].replace("Skin", "").strip().lower() for x in h if x.startswith("Good for")), None))
    # cross-check: claim says fragrance-free but INCI lists fragrance
    conflict = prod["claim_fragrance_free"] & (prod["has_fragrance"] == 1)
    prod["flag_claim_conflict"] = conflict
    log.add("features", "products claiming 'Fragrance Free' but listing fragrance in INCI flagged", int(conflict.sum()))
    log.add("features", f"{len(f['key_actives'])} active-ingredient flags + fragrance flag + 6 claim flags created", len(prod))

    # Merge into one review-level analysis table
    keep = ["product_id", "brand_name", "secondary_category", "tertiary_category", "price_usd", "size_ml",
            "price_per_ml", "price_per_ml_w", "loves_count", "loves_count_w", "n_ingredients", "n_key_actives",
            "has_fragrance", "good_for", "ingredients_missing", "highlights_missing", "size_missing", "price_imputed",
            "flag_price_outlier", "flag_claim_conflict", "limited_edition", "new", "online_only", "sephora_exclusive"]
    keep += [c for c in prod.columns if c.startswith(("has_", "claim_")) and c not in keep]
    df = rev.drop(columns=["product_name", "brand_name", "price_usd"], errors="ignore").merge(prod[keep], on="product_id", how="left")
    # reviewer-product fit
    df["skin_match"] = np.where(df["good_for"].isna() | (df["skin_type"] == "unknown"), np.nan,
                                (df["good_for"] == df["skin_type"]).astype(float))
    log.add("features", "skin_match feature (reviewer skin type == product 'Good for' claim)", int(df["skin_match"].notna().sum()))
    # Bayesian average rating per product: (C*m + sum ratings) / (m + n)
    m = f["bayesian_prior_reviews"]
    C = df["rating"].mean()
    g = df.groupby("product_id")["rating"].agg(["sum", "count"])
    bayes = (C * m + g["sum"]) / (m + g["count"])
    df["product_bayes_rating"] = df["product_id"].map(bayes)
    df["product_review_count"] = df["product_id"].map(g["count"])
    log.add("features", f"Bayesian-adjusted product rating (prior m = {m} reviews, global mean C = {C:.2f})", len(g))
    df["review_year"] = df["submission_time"].dt.year
    first = df.groupby("product_id")["submission_time"].transform("min")
    df["months_since_first_review"] = ((df["submission_time"] - first).dt.days / 30.44).round(1)
    log.add("features", "time features: review_year, months_since_first_review", int(df["submission_time"].notna().sum()))
    return df
