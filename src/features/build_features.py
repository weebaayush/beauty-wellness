"""Week 4 - feature engineering on the cleaned review table.

    python -m src.features.build_features

Reads  data/processed/reviews_clean.csv.gz   (output of the Week 3 pipeline)
Writes data/processed/features.csv.gz        (one row per review, all engineered features)
       reports/feature_catalogue.md          (name, family, definition, leakage-safe?)

Design rules
  * Every feature that uses other reviews looks only at reviews written EARLIER than the
    current one (expanding windows, shift(1)), so the same code is valid at prediction time.
  * Target-dependent encodings (brand target encoding) are fitted out-of-fold.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE: list[dict] = []


def register(name, family, definition, why):
    CATALOGUE.append({"feature": name, "family": family, "definition": definition, "why": why})


# ---------------------------------------------------------------- 1. customer behaviour

def customer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("submission_time").copy()
    profile = ["skin_type_missing", "skin_tone_missing", "eye_color_missing", "hair_color_missing"]
    df["profile_completeness"] = 4 - df[profile].astype(bool).sum(axis=1)
    register("profile_completeness", "Customer behaviour", "Number of the 4 optional profile fields the reviewer filled (0-4)",
             "Engaged shoppers disclose more; engagement may relate to satisfaction")
    g = df.groupby("author_hash")
    df["reviewer_prior_reviews"] = g.cumcount()
    register("reviewer_prior_reviews", "Customer behaviour", "Reviews the same reviewer wrote before this one",
             "Experienced reviewers are more critical / more trusted")
    prior_sum = g["rating"].cumsum() - df["rating"].fillna(0)
    df["reviewer_prior_mean_rating"] = (prior_sum / df["reviewer_prior_reviews"]).where(df["reviewer_prior_reviews"] > 0)
    register("reviewer_prior_mean_rating", "Customer behaviour", "Mean rating the reviewer gave in earlier reviews (NaN if first)",
             "Captures 'harsh' vs 'generous' reviewers")
    df["is_incentivised"] = df["flag_incentivised"].astype(int)
    register("is_incentivised", "Customer behaviour", "Review mentions a free / gifted product", "Incentivised reviews skew positive")
    return df


# ---------------------------------------------------------------- 2. time & market trends

def time_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["submission_time"]
    m = t.dt.month
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    register("month_sin / month_cos", "Market trend", "Cyclical encoding of review month",
             "Seasonality (e.g. sunscreen in summer, gifting in Nov-Dec) without a fake Dec->Jan jump")
    df["is_holiday_season"] = m.isin([11, 12]).astype(int)
    register("is_holiday_season", "Market trend", "Review written in November or December", "Gift purchases and sale events")
    first = df.groupby("product_id")["submission_time"].transform("min")
    df["days_since_launch"] = (t - first).dt.days
    df["is_early_adopter"] = (df["days_since_launch"] <= 90).astype(int)
    register("days_since_launch, is_early_adopter", "Market trend", "Days since product's first review; first 90 days flag",
             "Early buyers are enthusiasts; ratings often drift down after launch hype")
    return df


# ---------------------------------------------------------------- 3. product momentum (leak-free)

def product_history_features(df: pd.DataFrame, prior_m: int = 20) -> pd.DataFrame:
    df = df.sort_values("submission_time").copy()
    g = df.groupby("product_id")
    n_prior = g.cumcount()
    rating_filled = df["rating"].fillna(df["rating"].mean())
    sum_prior = g["rating"].transform(lambda s: s.fillna(s.mean()).cumsum().shift(1).fillna(0))
    C = df["rating"].mean()
    df["product_prior_reviews"] = n_prior
    df["product_prior_bayes_rating"] = (C * prior_m + sum_prior) / (prior_m + n_prior)
    register("product_prior_bayes_rating", "Product momentum",
             f"Bayesian mean of the product's ratings BEFORE this review (prior m={prior_m})",
             "Leak-free replacement for Week 3's product_bayes_rating, which used future reviews")
    rec_prior = g["is_recommended"].transform(lambda s: s.fillna(s.mean()).cumsum().shift(1).fillna(0))
    p0 = df["is_recommended"].mean()
    df["product_prior_rec_rate"] = (p0 * prior_m + rec_prior) / (prior_m + n_prior)
    register("product_prior_rec_rate", "Product momentum", "Smoothed recommendation rate before this review",
             "Social proof the shopper saw at purchase time")
    last10 = g["rating"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    df["product_momentum"] = last10 - df["product_prior_bayes_rating"]
    register("product_momentum", "Product momentum", "Mean of last 10 prior ratings minus long-run prior mean",
             "Detects products trending up (positive) or down (e.g. reformulation, batch issues)")
    span = g["submission_time"].transform(lambda s: (s - s.min()).dt.days + 30)
    df["review_velocity"] = (n_prior + 1) / (span / 30.44)
    register("review_velocity", "Product momentum", "Reviews per month up to this review", "Proxy for sales velocity / buzz")
    _ = rating_filled
    return df


# ---------------------------------------------------------------- 4. price positioning

def price_features(df: pd.DataFrame) -> pd.DataFrame:
    cat_med = df.groupby("secondary_category")["price_per_ml"].transform("median")
    df["price_rel_category"] = df["price_per_ml"] / cat_med
    register("price_rel_category", "Price positioning", "Price per mL / category median price per mL",
             "A $40 serum and $40 cleanser are not equally 'expensive'; relative price is comparable across categories")
    q = df.groupby("secondary_category")["price_per_ml"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=["budget", "mid", "premium", "luxury"]) if s.notna().sum() >= 8 else np.nan)
    df["price_tier"] = q.astype("string")
    register("price_tier", "Price positioning", "Quartile of price per mL within category", "Non-linear price effects; easy to explain")
    return df


# ---------------------------------------------------------------- 5. formulation

IRRITANTS = ["has_fragrance"]


def formulation_features(df: pd.DataFrame) -> pd.DataFrame:
    actives = [c for c in df.columns if c.startswith("has_") and c != "has_fragrance"]
    df["n_actives_present"] = df[actives].fillna(0).astype(int).sum(axis=1)
    df["strong_actives"] = df[["has_retinol", "has_salicylic_acid", "has_lactic_acid", "has_ascorbic_acid"]].fillna(0).astype(int).sum(axis=1)
    register("strong_actives", "Formulation", "Count of potentially irritating actives (retinol, salicylic, lactic, ascorbic acid)",
             "Powerful actives drive results but also irritation complaints")
    df["fragrance_free_verified"] = ((df["claim_fragrance_free"] == True) & (df["has_fragrance"] == 0)).astype(int)
    register("fragrance_free_verified", "Formulation", "Claims fragrance-free AND no fragrance in INCI",
             "Separates genuine from mislabelled claims")
    return df


# ---------------------------------------------------------------- 6. interactions (customer x product)

def interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    st = df["skin_type"]
    df["dry_x_fragrance"] = ((st == "dry") & (df["has_fragrance"] == 1)).astype(int)
    register("dry_x_fragrance", "Interaction", "Dry-skin reviewer AND product contains fragrance", "Directly tests H4")
    opposite = {"dry": "oily", "oily": "dry"}
    df["skin_mismatch"] = [int(g == opposite.get(s)) if isinstance(g, str) else 0 for s, g in zip(st, df["good_for"])]
    register("skin_mismatch", "Interaction", "Product targets the opposite skin type (dry vs oily)", "Strongest expected dissatisfaction case")
    df["oily_x_rich_texture"] = ((st == "oily") & df["tertiary_category"].isin(["Night Creams", "Face Oils"])).astype(int)
    register("oily_x_rich_texture", "Interaction", "Oily-skin reviewer AND night cream / face oil", "Texture-skin conflict")
    df["strong_actives_x_unknown_skin"] = df["strong_actives"] * (st == "unknown").astype(int)
    df["premium_x_first_purchase"] = ((df["price_tier"].isin(["premium", "luxury"])) & (df["reviewer_prior_reviews"] == 0)).astype(int)
    register("premium_x_first_purchase", "Interaction", "Premium/luxury product AND reviewer's first review",
             "High expectations of first-time premium buyers")
    return df


# ---------------------------------------------------------------- 7. text (lexicon-based, explainable)

ASPECTS = {
    "aspect_skin_reaction_neg": r"broke me out|breakout|burn|sting|redness|itch|rash|irritat|allerg",
    "aspect_texture": r"texture|greasy|sticky|lightweight|absorb|pill|thick|creamy|film",
    "aspect_scent": r"smell|scent|fragrance|perfume|odou?r",
    "aspect_value": r"expensive|price|worth|cheap|value|penny|size",
    "aspect_packaging": r"pump|packag|bottle|jar|tube|leak",
    "aspect_efficacy": r"brighter|dark spots|results|worked|did nothing|hydrated|plump|clear",
}


def text_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["review_text_clean"].fillna("")
    for name, pat in ASPECTS.items():
        df[name] = t.str.count(pat, flags=re.I)
    register("aspect_* (6)", "Text", "Counts of aspect keywords: skin reaction, texture, scent, value, packaging, efficacy",
             "Explainable first version of the aspect analysis in H3 (replaced by a transformer model later)")
    df["exclamations"] = t.str.count("!")
    df["has_emoji"] = t.str.contains(r"[\U0001F300-\U0001FAFF☀-➿]", regex=True).astype(int)
    df["caps_ratio"] = t.apply(lambda s: sum(c.isupper() for c in s) / max(1, sum(c.isalpha() for c in s)))
    register("exclamations, has_emoji, caps_ratio", "Text", "Emotional intensity markers", "Strong emotion -> extreme ratings")
    return df


# ---------------------------------------------------------------- 8. encodings

def brand_target_encoding(df: pd.DataFrame, k: int = 5, smoothing: int = 50, seed: int = 42) -> pd.DataFrame:
    """Out-of-fold, smoothed mean of is_recommended per brand: no row sees its own label."""
    df = df.copy()
    y = df["is_recommended"]
    prior = y.mean()
    enc = pd.Series(np.nan, index=df.index)
    for tr, va in KFold(k, shuffle=True, random_state=seed).split(df):
        stats = df.iloc[tr].groupby("brand_name")["is_recommended"].agg(["sum", "count"])
        te = (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)
        enc.iloc[va] = df.iloc[va]["brand_name"].map(te).fillna(prior).values
    df["brand_te"] = enc
    df["brand_freq"] = df["brand_name"].map(df["brand_name"].value_counts(normalize=True))
    register("brand_te", "Encoding", f"Out-of-fold ({k}-fold) smoothed recommendation rate of the brand (m={smoothing})",
             "Captures brand reputation in one column instead of 30 one-hot columns, without target leakage")
    register("brand_freq", "Encoding", "Share of reviews belonging to the brand", "Brand size / visibility")
    return df


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["submission_time"] = pd.to_datetime(df["submission_time"])
    df = df[df["submission_time"].notna()]
    for fn in [customer_features, time_features, product_history_features, price_features,
               formulation_features, interaction_features, text_features, brand_target_encoding]:
        df = fn(df)
    return df.sort_values("submission_time").reset_index(drop=True)


if __name__ == "__main__":
    src = ROOT / "data" / "processed" / "reviews_clean.csv.gz"
    feats = build(pd.read_csv(src, low_memory=False))
    feats.to_csv(ROOT / "data" / "processed" / "features.csv.gz", index=False, compression="gzip")
    cat = pd.DataFrame(CATALOGUE)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "feature_catalogue.md").write_text(
        "# Feature catalogue (Week 4)\n\n| Feature | Family | Definition | Why |\n|---|---|---|---|\n" +
        "\n".join(f"| {r.feature} | {r.family} | {r.definition} | {r.why} |" for r in cat.itertuples()), encoding="utf-8")
    print(feats.shape, "features written;", len(cat), "feature groups catalogued")
