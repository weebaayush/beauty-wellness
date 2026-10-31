"""Normalisation / encoding for modelling, built so it is fitted on the training split only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

SKEWED = ["price_per_ml_w", "loves_count_w", "review_word_count_w", "product_review_count"]   # log1p + robust scale
SYMMETRIC = ["n_ingredients", "n_key_actives", "months_since_first_review"]                   # standard scale
BOUNDED = ["helpfulness_smoothed", "product_bayes_rating"]                                     # min-max to [0, 1]
CATEGORICAL = ["skin_type", "skin_tone", "secondary_category", "brand_name"]
BINARY_PREFIX = ("has_", "claim_", "flag_incentivised", "skin_type_missing", "skin_tone_missing",
                 "ingredients_missing", "price_imputed", "sephora_exclusive", "online_only", "new", "limited_edition")


def binary_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(BINARY_PREFIX) and c != "has_fragrance"] + ["has_fragrance", "skin_match"]


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    log_robust = Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                           ("scale", RobustScaler())])
    std = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    mm = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", MinMaxScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False))])
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    return ColumnTransformer([("skewed", log_robust, SKEWED), ("symmetric", std, SYMMETRIC), ("bounded", mm, BOUNDED),
                              ("categorical", cat, CATEGORICAL), ("binary", binary, binary_columns(df))],
                             verbose_feature_names_out=True)


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Train on older reviews, test on the newest 20% - mimics predicting future reviews and avoids leakage."""
    d = df[df["target_available"] & df["submission_time"].notna()].sort_values("submission_time")
    cut = int(len(d) * (1 - test_frac))
    return d.iloc[:cut], d.iloc[cut:]


def prepare_model_matrix(df: pd.DataFrame):
    train, test = time_split(df)
    for part in (train, test):
        for c in binary_columns(df):
            part[c] = part[c].astype(float)
    pre = build_preprocessor(df)
    X_train = pre.fit_transform(train)          # fit ONLY on training data
    X_test = pre.transform(test)                # apply the same learned parameters
    names = pre.get_feature_names_out()
    return (pd.DataFrame(X_train, columns=names, index=train.index), train["is_recommended"].astype(int),
            pd.DataFrame(X_test, columns=names, index=test.index), test["is_recommended"].astype(int), pre)
