"""Cleaning steps for the beauty & wellness review data.

Each function takes a DataFrame (plus config and a CleaningLog), returns a new
DataFrame and records what it changed. Steps never modify their input in place,
so every stage can be inspected and tested on its own.
"""
from __future__ import annotations

import ast
import hashlib
import html
import re

import numpy as np
import pandas as pd

from .log import CleaningLog

# ---------------------------------------------------------------- helpers

def _norm_key(s: str) -> str:
    """Lower-case, trim and collapse whitespace - used before every mapping lookup."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def build_lookup(mapping: dict) -> dict:
    """{canonical: [variants]} -> {variant_key: canonical} (canonical maps to itself)."""
    lut = {}
    for canon, variants in mapping.items():
        lut[_norm_key(canon)] = canon
        for v in variants:
            lut[_norm_key(v)] = canon
    return lut


# ---------------------------------------------------------------- 1. ingestion & schema

REQUIRED_PRODUCT_COLS = ["product_id", "brand_name", "price_usd", "secondary_category", "ingredients",
                         "highlights", "size", "loves_count"]
REQUIRED_REVIEW_COLS = ["author_id", "product_id", "rating", "is_recommended", "review_text",
                        "submission_time", "skin_type", "skin_tone", "helpfulness",
                        "total_feedback_count", "total_pos_feedback_count", "total_neg_feedback_count"]


def check_schema(df: pd.DataFrame, required: list[str], name: str, log: CleaningLog) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")
    log.add("schema", f"{name}: all {len(required)} required columns present", 0)


# ---------------------------------------------------------------- 2. standardise formats

PRICE_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def parse_price(x) -> float:
    """'$42.00', 'USD 42', '42.00 ' -> 42.0; unparseable -> NaN."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    m = PRICE_RE.search(str(x).replace(",", ""))
    return float(m.group()) if m else np.nan


SIZE_ML_RE = re.compile(r"(\d+(?:\.\d+)?)\s*ml", re.I)
SIZE_OZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:fl\.?\s*)?oz", re.I)
OZ_TO_ML = 29.5735


def parse_size_ml(x) -> float:
    """Prefer the mL value when both are given ('1.7 oz/ 50 mL'); else convert oz."""
    if pd.isna(x):
        return np.nan
    s = str(x)
    m = SIZE_ML_RE.search(s)
    if m:
        return float(m.group(1))
    m = SIZE_OZ_RE.search(s)
    return round(float(m.group(1)) * OZ_TO_ML, 1) if m else np.nan


def canonical_brand(b) -> str | float:
    """' LUMEA LABS. ' / 'Petal and Pine' -> 'lumea labs' / 'petal & pine' (a match key)."""
    if pd.isna(b):
        return np.nan
    s = _norm_key(b).rstrip(".")
    s = re.sub(r"\band\b", "&", s)
    return re.sub(r"\s+", " ", s)


def standardise_products(df: pd.DataFrame, cfg: dict, log: CleaningLog) -> pd.DataFrame:
    out = df.copy()
    # price
    before_text = (out["price_usd"].notna() & ~out["price_usd"].astype(str).str.fullmatch(r"-?\d+(?:\.\d+)?")).sum()
    out["price_usd"] = out["price_usd"].apply(parse_price)
    log.add("standardise", "price strings parsed to float ($, 'USD', spaces removed)", int(before_text))
    # brand: build canonical key, then use the most frequent original spelling as display name
    out["brand_key"] = out["brand_name"].apply(canonical_brand)
    display = (out.groupby("brand_key")["brand_name"]
               .agg(lambda s: s.str.strip().value_counts().index[0]))
    changed = (out["brand_name"] != out["brand_key"].map(display)).sum()
    out["brand_name"] = out["brand_key"].map(display)
    log.add("standardise", "brand names unified via canonical key", int(changed))
    # category
    lut = build_lookup(cfg["mappings"]["secondary_category"])
    new = out["secondary_category"].map(lambda c: lut.get(_norm_key(c), c) if pd.notna(c) else c)
    log.add("standardise", "category spellings mapped to 6 canonical categories",
            int((new != out["secondary_category"]).sum()))
    out["secondary_category"] = new
    # size
    out["size_ml"] = out["size"].apply(parse_size_ml)
    log.add("standardise", "size text parsed to millilitres (oz x 29.5735)", int(out["size_ml"].notna().sum()))
    # highlights: string repr of list -> python list
    def to_list(h):
        if pd.isna(h):
            return []
        try:
            v = ast.literal_eval(h) if isinstance(h, str) else h
            return [str(i).strip() for i in v]
        except (ValueError, SyntaxError):
            return [p.strip() for p in str(h).strip("[]").split(",") if p.strip()]
    out["highlights_list"] = out["highlights"].apply(to_list)
    log.add("standardise", "highlights parsed from text to lists", int(out["highlights"].notna().sum()))
    return out


def _parse_dates(s: pd.Series) -> pd.Series:
    """Try each known format in turn; anything left is NaT."""
    s = s.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%Y-%m-%dT%H:%M:%S"]:
        todo = out.isna() & s.notna()
        if not todo.any():
            break
        out.loc[todo] = pd.to_datetime(s[todo], format=fmt, errors="coerce")
    return out


def standardise_reviews(df: pd.DataFrame, cfg: dict, log: CleaningLog) -> pd.DataFrame:
    out = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore").copy()
    # rating: '5 stars' -> 5
    txt = (~out["rating"].astype(str).str.fullmatch(r"\d+(?:\.0)?|nan")).sum()
    out["rating"] = pd.to_numeric(out["rating"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    log.add("standardise", "rating text ('5 stars') converted to numbers", int(txt))
    # is_recommended: Yes/No/1/0/1.0 -> 1/0
    rec_map = {"1": 1, "1.0": 1, "yes": 1, "true": 1, "0": 0, "0.0": 0, "no": 0, "false": 0}
    raw = out["is_recommended"]
    out["is_recommended"] = raw.map(lambda v: rec_map.get(_norm_key(v)) if pd.notna(v) else np.nan).astype("Float64")
    log.add("standardise", "is_recommended Yes/No/1.0 values unified to 1/0",
            int(raw.astype(str).str.lower().isin(["yes", "no"]).sum()))
    # skin type / tone
    for col in ["skin_type", "skin_tone"]:
        lut = build_lookup(cfg["mappings"][col])
        before = out[col].copy()
        out[col] = out[col].map(lambda v: lut.get(_norm_key(v), np.nan) if pd.notna(v) else np.nan)
        changed = (before.notna() & (before != out[col])).sum()
        log.add("standardise", f"{col} variants mapped to {len(cfg['mappings'][col])} canonical labels", int(changed))
    for col in ["eye_color", "hair_color"]:
        out[col] = out[col].map(lambda v: _norm_key(v) if pd.notna(v) else np.nan)
    # dates
    parsed = _parse_dates(out["submission_time"])
    non_iso = (~out["submission_time"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")).sum()
    out["submission_time"] = parsed
    log.add("standardise", "dates in 4 formats parsed to one datetime type", int(non_iso))
    out["author_id"] = out["author_id"].astype(str)
    return out


# ---------------------------------------------------------------- 3. validity rules

def apply_validity_rules(prod: pd.DataFrame, rev: pd.DataFrame, cfg: dict, log: CleaningLog):
    v = cfg["validity"]
    prod, rev = prod.copy(), rev.copy()
    bad = ~prod["price_usd"].between(v["price_min_usd"], v["price_max_usd"]) & prod["price_usd"].notna()
    prod.loc[bad, "price_usd"] = np.nan
    log.add("validity", f"prices outside ${v['price_min_usd']}-${v['price_max_usd']} set to missing", int(bad.sum()))
    neg = prod["loves_count"] < 0
    prod.loc[neg, "loves_count"] = prod.loc[neg, "loves_count"].abs()
    log.add("validity", "negative loves_count corrected (sign error -> absolute value)", int(neg.sum()))

    bad = ~rev["rating"].between(v["rating_min"], v["rating_max"]) & rev["rating"].notna()
    rev.loc[bad, "rating"] = np.nan
    log.add("validity", "ratings outside 1-5 set to missing", int(bad.sum()))
    # helpfulness stored as percentage (e.g. 67 instead of 0.67)
    pct = rev["helpfulness"] > 1
    rev.loc[pct, "helpfulness"] = rev.loc[pct, "helpfulness"] / 100
    log.add("validity", "helpfulness on 0-100 scale rescaled to 0-1", int(pct.sum()))
    # feedback counts must add up: total = positive + negative
    calc = rev["total_pos_feedback_count"] + rev["total_neg_feedback_count"]
    inc = rev["total_feedback_count"] != calc
    rev.loc[inc, "total_feedback_count"] = calc[inc]
    log.add("validity", "total_feedback_count recomputed as pos + neg where inconsistent", int(inc.sum()))
    # dates
    too_late = rev["submission_time"] > pd.Timestamp(v["max_date"])
    too_early = rev["submission_time"] < pd.Timestamp(v["min_date"])
    unparsed = rev["submission_time"].isna()
    rev.loc[too_late | too_early, "submission_time"] = pd.NaT
    log.add("validity", f"dates after snapshot ({v['max_date']}) or before {v['min_date']} set to missing",
            int((too_late | too_early).sum()))
    log.add("validity", "unparseable dates left as missing", int(unparsed.sum()))
    # logical conflict: 1-2 stars but recommended -> keep, but flag for review
    rev["flag_rating_rec_conflict"] = (rev["rating"] <= 1) & (rev["is_recommended"] == 1)
    log.add("validity", "1-star reviews marked 'recommended' flagged as conflicts", int(rev["flag_rating_rec_conflict"].sum()))
    return prod, rev


# ---------------------------------------------------------------- 4. duplicates & integrity

def remove_duplicates(prod: pd.DataFrame, rev: pd.DataFrame, log: CleaningLog):
    n = len(prod)
    prod = prod.drop_duplicates(subset="product_id", keep="first")
    log.add("duplicates", "duplicate product rows removed", n - len(prod))
    rev = rev.copy()
    rev["_text_key"] = rev["review_text"].fillna("").str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    n = len(rev)
    rev = rev.drop_duplicates(subset=["author_id", "product_id", "_text_key"], keep="first")
    log.add("duplicates", "exact duplicate reviews removed (same author, product, text)", n - len(rev))
    # syndicated: same author + same non-trivial text on a *different* product -> keep first only
    long_text = rev["_text_key"].str.split().str.len() >= 5
    synd = long_text & rev.duplicated(subset=["author_id", "_text_key"], keep="first")
    rev = rev[~synd]
    log.add("duplicates", "syndicated copies removed (same author + text on another product)", int(synd.sum()))
    return prod, rev.drop(columns="_text_key")


def enforce_integrity(prod: pd.DataFrame, rev: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    orphan = ~rev["product_id"].isin(prod["product_id"])
    log.add("integrity", "reviews whose product_id has no product record removed", int(orphan.sum()))
    return rev[~orphan].copy()


# ---------------------------------------------------------------- 5. text cleaning

MOJIBAKE = {"â€™": "'", "â€˜": "'", "â€œ": '"', "â€\x9d": '"', "â€“": "-", "â€”": "-", "Ã©": "é", "â€¦": "..."}
TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def clean_text(t) -> str:
    if pd.isna(t):
        return ""
    s = str(t)
    for bad, good in MOJIBAKE.items():
        s = s.replace(bad, good)
    s = html.unescape(s)
    s = TAG_RE.sub(" ", s)
    s = URL_RE.sub(" [URL] ", s)
    s = EMAIL_RE.sub(" [EMAIL] ", s)       # PII scrubbing
    return re.sub(r"\s+", " ", s).strip()  # emojis are kept on purpose: they carry sentiment


def ascii_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return 1.0 if not letters else sum(c.isascii() for c in letters) / len(letters)


SPANISH_FRENCH = re.compile(r"\b(muy|pero|esta|todos|los|días|je|très|produit|recommande|la|el|uso)\b", re.I)


def clean_review_text(rev: pd.DataFrame, cfg: dict, log: CleaningLog) -> pd.DataFrame:
    t = cfg["text"]
    rev = rev.copy()
    raw = rev["review_text"].fillna("")
    for name, pat in [("HTML tags/entities", r"<[^>]+>|&\w+;"), ("URLs", URL_RE.pattern),
                      ("e-mail addresses (PII)", EMAIL_RE.pattern), ("mojibake characters", "â€")]:
        log.add("text", f"{name} cleaned", int(raw.str.contains(pat, regex=True).sum()))
    rev["review_text_clean"] = raw.apply(clean_text)
    rev["review_word_count"] = rev["review_text_clean"].str.split().str.len().fillna(0).astype(int)
    rev["flag_short_text"] = rev["review_word_count"] < t["min_words_for_nlp"]
    log.add("text", f"reviews with < {t['min_words_for_nlp']} words flagged (excluded from NLP only)", int(rev["flag_short_text"].sum()))
    # language heuristic: low ASCII share or common Spanish/French function words
    sf_hits = rev["review_text_clean"].str.count(SPANISH_FRENCH.pattern, flags=re.I)
    rev["flag_non_english"] = (rev["review_text_clean"].apply(ascii_ratio) < t["min_ascii_ratio"]) | (sf_hits >= 2)
    log.add("text", "possibly non-English reviews flagged", int(rev["flag_non_english"].sum()))
    pat = "|".join(t["incentive_patterns"])
    rev["flag_incentivised"] = rev["review_text_clean"].str.contains(pat, case=False, regex=True)
    log.add("text", "incentivised reviews flagged (free product / Influenster)", int(rev["flag_incentivised"].sum()))
    # hash author_id: keep the ability to group by author without storing the raw ID
    rev["author_hash"] = rev["author_id"].map(lambda a: hashlib.sha256(f"bw-salt:{a}".encode()).hexdigest()[:12])
    rev = rev.drop(columns=["author_id"])
    log.add("text", "author_id replaced by salted SHA-256 hash (privacy)", len(rev))
    return rev


# ---------------------------------------------------------------- 6. ingredients

DISCLAIMER_RE = re.compile(r"please be aware that ingredient lists may change.*$", re.I | re.S)
DESC_RE = re.compile(r"-[^:]{2,40}:[^.]*\.\s*")  # '-Niacinamide: Minimizes pores. ' marketing prefix


def split_inci(s: str) -> list[str]:
    """Split on commas that are not inside parentheses."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p for p in parts]


def parse_ingredients(raw, synonyms: dict) -> list[str]:
    if pd.isna(raw):
        return []
    s = DISCLAIMER_RE.sub("", str(raw))
    s = DESC_RE.sub("", s)
    out = []
    for p in split_inci(s):
        k = _norm_key(p).strip(" .;")
        if not k:
            continue
        out.append(synonyms.get(k, k))
    return list(dict.fromkeys(out))  # de-duplicate, keep INCI order (order ~ concentration)


def clean_ingredients(prod: pd.DataFrame, cfg: dict, log: CleaningLog) -> pd.DataFrame:
    prod = prod.copy()
    syn = build_lookup(cfg["mappings"]["ingredient_synonyms"])
    raw = prod["ingredients"].fillna("")
    log.add("ingredients", "legal disclaimer text removed", int(raw.str.contains("Please be aware", case=False).sum()))
    log.add("ingredients", "marketing descriptions ('-Niacinamide: ...') removed", int(raw.str.contains(r"^-[^:]{2,40}:", regex=True).sum()))
    prod["ingredient_list"] = prod["ingredients"].apply(lambda r: parse_ingredients(r, syn))
    variants = sum(1 for lst, r in zip(prod["ingredient_list"], raw)
                   for k in [_norm_key(x).strip(" .") for x in split_inci(DESC_RE.sub("", DISCLAIMER_RE.sub("", r)))]
                   if k in syn and syn[k] != k)
    log.add("ingredients", "ingredient synonyms unified (e.g. Aqua -> water, Parfum -> fragrance)", int(variants))
    prod["n_ingredients"] = prod["ingredient_list"].str.len()
    return prod
