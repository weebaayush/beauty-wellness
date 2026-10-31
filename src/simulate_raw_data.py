"""Week 3 - simulate a raw, messy Sephora-style skincare dataset.

Why simulate? The real Sephora dataset (Kaggle) has the same schema, but to
*test* a cleaning pipeline we need data where we know exactly which problems
were injected and how many. Every issue below mirrors a problem that occurs in
public beauty & wellness data. Rates are documented in INJECTED_ISSUES and
written to data/raw/simulated/injection_log.json so the pipeline's results can
be checked against ground truth.

Usage:
    python src/simulate_raw_data.py --products 1200 --reviews 60000 --seed 42
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "simulated"

# Fictional brands (no real brand data is fabricated)
BRANDS = ["Lumea Labs", "Verde Botanics", "Skinsmith", "Aurora Derm", "Mira Clean", "Koa Rituals",
          "Haven Skin", "Petal & Pine", "Nova Bio", "Saffron Lane", "Glacier Lab", "Oasis Dew",
          "Terra Nova", "Pure Atelier", "Indigo Field", "Calm Theory", "Bloom Chemistry", "Ember & Oak",
          "Sora Skin", "Velvet Science", "Crescent Care", "Juniper Lab", "Ivory Coast Co", "Rainleaf",
          "Aster Rx", "Moss & Mineral", "Halo Derma", "Kintsugi Skin", "Prism Botanica", "Daybreak Lab"]
CATEGORIES = {
    "Moisturizers": ["Moisturizers", "Face Oils", "Night Creams", "Mists & Essences"],
    "Treatments": ["Face Serums", "Facial Peels", "Blemish & Acne Treatments"],
    "Cleansers": ["Face Wash & Cleansers", "Exfoliators", "Toners", "Makeup Removers"],
    "Sunscreen": ["Face Sunscreen", "Body Sunscreen"],
    "Eye Care": ["Eye Creams & Treatments", "Eye Masks"],
    "Masks": ["Face Masks", "Sheet Masks"],
}
CATEGORY_VARIANTS = {  # inconsistent spellings seen in scraped / merged data
    "Moisturizers": ["moisturisers", "Moisturizer", "MOISTURIZERS"],
    "Treatments": ["treatments", "Treatment"],
    "Cleansers": ["cleansers", "Cleanser"],
    "Sunscreen": ["Sunscreens", "sun screen"],
    "Eye Care": ["Eye care", "eye-care"],
    "Masks": ["masks", "Mask"],
}
BASE_INGREDIENTS = ["Water", "Glycerin", "Butylene Glycol", "Dimethicone", "Squalane", "Cetearyl Alcohol",
                    "Caprylic/Capric Triglyceride", "Phenoxyethanol", "Ethylhexylglycerin", "Xanthan Gum",
                    "Sodium Hyaluronate", "Niacinamide", "Panthenol", "Tocopherol", "Allantoin",
                    "Ceramide NP", "Retinol", "Ascorbic Acid", "Salicylic Acid", "Zinc Oxide",
                    "Centella Asiatica Extract", "Shea Butter", "Alcohol Denat.", "Lactic Acid"]
FRAGRANCE = ["Fragrance", "Linalool", "Limonene", "Citronellol"]
SYNONYM_NOISE = {"Water": ["Aqua", "Water/Aqua/Eau", "Aqua (Water)", "WATER"],
                 "Fragrance": ["Parfum", "Fragrance (Parfum)", "fragrance"],
                 "Shea Butter": ["Butyrospermum Parkii (Shea) Butter"],
                 "Sodium Hyaluronate": ["Hyaluronic Acid"]}
DISCLAIMER = ("Please be aware that ingredient lists may change or vary from time to time. "
              "Please refer to the ingredient list on the product package you receive for the most "
              "up to date list of ingredients.")
HIGHLIGHT_POOL = ["Vegan", "Clean at Sephora", "Fragrance Free", "Hydrating", "Good for: Dry Skin",
                  "Good for: Oily Skin", "Good for: Combination Skin", "Good for: Sensitive Skin",
                  "Without Parabens", "Hyaluronic Acid", "Cruelty-Free", "Oil Free"]

SKIN_TYPES = ["dry", "oily", "combination", "normal"]
SKIN_TYPE_VARIANTS = {"dry": ["Dry", " DRY", "dry skin"], "oily": ["Oily", "oily ", "OILY"],
                      "combination": ["combo", "Combination", "comb."], "normal": ["Normal", "normal "]}
SKIN_TONES = ["fair", "light", "lightMedium", "fairLight", "medium", "mediumTan", "tan", "deep", "rich",
              "olive", "porcelain", "dark"]
TONE_P = [.08, .2, .17, .1, .15, .1, .08, .04, .02, .03, .02, .01]
TONE_VARIANTS = {"lightMedium": ["light medium", "Light-Medium"], "mediumTan": ["medium tan", "Medium-Tan"],
                 "fairLight": ["fair light"]}

ASPECT_PHRASES = {
    "pos": ["absorbs quickly", "my skin feels so hydrated", "holy grail product", "lovely lightweight texture",
            "noticeably brighter skin after two weeks", "no white cast", "gentle on my sensitive skin",
            "worth every penny", "the packaging is beautiful", "smells amazing"],
    "neg": ["broke me out within a week", "left a greasy film", "burning sensation on application",
            "way too expensive for the size", "the pump broke after a few days", "smells way too strong",
            "pilled under my makeup", "did nothing for my dark spots", "caused redness and itching",
            "dried out my skin"],
}
FOREIGN = ["Me encanta esta crema, la uso todos los días.", "Produit très doux, je recommande.",
           "Muy buena hidratación pero el precio es alto."]
INCENTIVE = ["I received this product for free in exchange for my honest review.",
             "Received complimentary from Influenster."]


def _ingredients(rng, fragranced):
    n = rng.integers(8, 20)
    ings = ["Water"] + list(rng.choice(BASE_INGREDIENTS[1:], size=n - 1, replace=False))
    if fragranced:
        ings += list(rng.choice(FRAGRANCE, size=rng.integers(1, 3), replace=False))
    return ings


def make_products(rng, n):
    rows = []
    for i in range(n):
        primary = rng.choice(["Moisturizers", "Treatments", "Cleansers", "Sunscreen", "Eye Care", "Masks"],
                             p=[.3, .25, .2, .1, .08, .07])
        primary = str(primary)
        secondary = str(rng.choice(CATEGORIES[primary]))
        brand = str(rng.choice(BRANDS))
        fragranced = rng.random() < 0.35
        ings = _ingredients(rng, fragranced)
        good_for = rng.choice(["Dry", "Oily", "Combination", "Sensitive", None], p=[.25, .2, .15, .15, .25])
        hl = list(rng.choice([h for h in HIGHLIGHT_POOL if not h.startswith("Good for")], size=rng.integers(1, 4),
                             replace=False))
        if good_for:
            hl.append(f"Good for: {good_for} Skin")
        if rng.random() < (0.6 if not fragranced else 0.06):
            hl.append("Fragrance Free")
        hl = list(dict.fromkeys(str(h) for h in hl))
        if fragranced and "Fragrance Free" in hl and rng.random() > 0.08:
            hl.remove("Fragrance Free")   # ~8% keep a wrong claim -> label errors exist in real data too
        ml = float(rng.choice([15, 30, 50, 75, 100, 150, 200]))
        price = round(float(np.exp(rng.normal(3.6, 0.5)) * (ml / 50) ** 0.5), 0)
        quality = rng.normal(0, 0.6)  # latent product quality
        rows.append(dict(product_id=f"P{400000 + i}", product_name=f"{brand.split()[0]} {secondary.split()[0]} No.{i}",
                         brand_id=BRANDS.index(brand) + 1000, brand_name=brand, primary_category="Skincare",
                         secondary_category=primary, tertiary_category=secondary, size_ml=ml, price_usd=price,
                         ingredients=ings, highlights=hl, fragranced=fragranced, good_for=good_for,
                         quality=quality, loves_count=int(rng.lognormal(8, 1.5)),
                         limited_edition=int(rng.random() < .05), new=int(rng.random() < .08),
                         online_only=int(rng.random() < .2), sephora_exclusive=int(rng.random() < .25)))
    return pd.DataFrame(rows)


def make_reviews(rng, products, n):
    p_idx = rng.choice(len(products), size=n, p=_popularity(rng, len(products)))
    rows = []
    start = pd.Timestamp("2016-01-01").value // 10**9
    end = pd.Timestamp("2023-03-15").value // 10**9
    for k, pi in enumerate(p_idx):
        p = products.iloc[pi]
        skin = rng.choice(SKIN_TYPES, p=[.25, .2, .4, .15])
        tone = rng.choice(SKIN_TONES, p=TONE_P)
        # latent satisfaction: product quality + skin match + fragrance penalty for dry skin
        gf = p.good_for if isinstance(p.good_for, str) else ""
        match = 0.5 if gf.lower() == skin else (-0.4 if gf in ("Dry", "Oily") else 0)
        frag_pen = -0.35 if (p.fragranced and skin == "dry") else 0
        latent = 1.3 + p.quality + match + frag_pen + rng.normal(0, 1.0)
        rating = int(np.clip(np.round(3 + latent), 1, 5))
        rec = int(rating >= 4 or (rating == 3 and rng.random() < .4))
        aspects = list(rng.choice(ASPECT_PHRASES["pos" if rating >= 4 else "neg"], size=rng.integers(1, 3), replace=False))
        if rating == 3:
            aspects = [rng.choice(ASPECT_PHRASES["pos"]), rng.choice(ASPECT_PHRASES["neg"])]
        text = f"{'Love it' if rating >= 4 else 'Not for me' if rating <= 2 else 'It is okay'}, " + ", ".join(aspects) + "."
        total_fb = int(rng.poisson(1.5) if rng.random() < .6 else 0)
        pos_fb = int(rng.binomial(total_fb, .8 if rating <= 2 else .7)) if total_fb else 0
        rows.append(dict(author_id=str(int(rng.integers(1e9, 9.9e9))), rating=rating, is_recommended=rec,
                         helpfulness=(pos_fb / total_fb) if total_fb else np.nan, total_feedback_count=total_fb,
                         total_neg_feedback_count=total_fb - pos_fb, total_pos_feedback_count=pos_fb,
                         submission_time=pd.Timestamp(int(rng.integers(start, end)), unit="s").strftime("%Y-%m-%d"),
                         review_text=text, review_title=rng.choice(["Love it", "Meh", "Great!", "Not for me", None]),
                         skin_tone=tone, eye_color=rng.choice(["brown", "blue", "green", "hazel", "gray"]),
                         skin_type=skin, hair_color=rng.choice(["black", "brown", "blonde", "red", "gray"]),
                         product_id=p.product_id, product_name=p.product_name, brand_name=p.brand_name,
                         price_usd=p.price_usd))
    return pd.DataFrame(rows)


def _popularity(rng, n):
    w = rng.pareto(1.2, n) + 0.05
    return w / w.sum()


def pick(rng, n, rate):
    return rng.random(n) < rate


def inject_product_issues(rng, prod, log):
    n = len(prod)
    out = prod.copy()
    # ingredients → raw Sephora-style string
    def fmt(ings):
        ings = [rng.choice(SYNONYM_NOISE[i]) if i in SYNONYM_NOISE and rng.random() < .5 else i for i in ings]
        if rng.random() < .3:
            ings = [i.upper() if rng.random() < .5 else i.lower() for i in ings]
        s = ", ".join(ings) + "."
        if rng.random() < .4:
            s = "-Niacinamide: Minimizes the look of pores. " + s
        if rng.random() < .5:
            s += " " + DISCLAIMER
        return s
    out["ingredients"] = out["ingredients"].apply(fmt)
    m = pick(rng, n, .06); out.loc[m, "ingredients"] = np.nan; log["product_ingredients_missing"] = int(m.sum())
    out["highlights"] = out["highlights"].apply(lambda h: str(h))
    m = pick(rng, n, .25); out.loc[m, "highlights"] = np.nan; log["product_highlights_missing"] = int(m.sum())
    # size text in mixed formats
    def size_txt(ml):
        oz = round(ml / 29.5735, 1)
        return rng.choice([f"{oz} oz/ {int(ml)} mL", f"{int(ml)}ml", f"{oz} oz", f"{int(ml)} ML", f"{oz}oz / {int(ml)}mL"])
    out["size"] = out["size_ml"].apply(size_txt)
    m = pick(rng, n, .15); out.loc[m, "size"] = np.nan; log["product_size_missing"] = int(m.sum())
    # brand name inconsistencies
    m = pick(rng, n, .08)
    out.loc[m, "brand_name"] = out.loc[m, "brand_name"].apply(
        lambda b: rng.choice([b.upper(), b.lower(), f" {b} ", b + ".", b.replace("&", "and")]))
    log["product_brand_inconsistent"] = int(m.sum())
    # category spelling variants
    m = pick(rng, n, .10)
    out.loc[m, "secondary_category"] = out.loc[m, "secondary_category"].apply(lambda c: rng.choice(CATEGORY_VARIANTS[c]))
    log["product_category_variant"] = int(m.sum())
    # price issues
    out["price_usd"] = out["price_usd"].astype(object)
    m = pick(rng, n, .05)
    out.loc[m, "price_usd"] = out.loc[m, "price_usd"].apply(lambda p: rng.choice([f"${p:.2f}", f"USD {p:.0f}", f"{p:.2f} "]))
    log["product_price_as_text"] = int(m.sum())
    m = pick(rng, n, .02) & out["price_usd"].notna(); out.loc[m, "price_usd"] = np.nan; log["product_price_missing"] = int(m.sum())
    m = pick(rng, n, .006); out.loc[m, "price_usd"] = rng.choice([0, 9999, -15], size=m.sum()); log["product_price_invalid"] = int(m.sum())
    m = pick(rng, n, .006) & out["price_usd"].apply(lambda v: isinstance(v, float) and v > 0)
    out.loc[m, "price_usd"] = out.loc[m, "price_usd"].astype(float) * 8; log["product_price_extreme_luxury"] = int(m.sum())
    m = pick(rng, n, .01); out.loc[m, "loves_count"] = -out.loc[m, "loves_count"]; log["product_loves_negative"] = int(m.sum())
    out = out.drop(columns=["size_ml", "fragranced", "good_for", "quality"])
    dup = out.sample(frac=.01, random_state=1); log["product_duplicate_rows"] = len(dup)
    return pd.concat([out, dup], ignore_index=True)


def inject_review_issues(rng, rev, log):
    n = len(rev)
    out = rev.copy()
    for col, rate in [("skin_type", .30), ("skin_tone", .25), ("eye_color", .35), ("hair_color", .30)]:
        m = pick(rng, n, rate); out.loc[m, col] = np.nan; log[f"review_{col}_missing"] = int(m.sum())
    m = pick(rng, n, .10) & out.skin_type.notna()
    out.loc[m, "skin_type"] = out.loc[m, "skin_type"].apply(lambda s: rng.choice(SKIN_TYPE_VARIANTS[s]))
    log["review_skin_type_variant"] = int(m.sum())
    m = pick(rng, n, .08) & out.skin_tone.isin(TONE_VARIANTS)
    out.loc[m, "skin_tone"] = out.loc[m, "skin_tone"].apply(lambda s: rng.choice(TONE_VARIANTS[s]))
    m2 = pick(rng, n, .02); out.loc[m2, "skin_tone"] = "notSureST"
    log["review_skin_tone_variant"] = int(m.sum()); log["review_skin_tone_notsure"] = int(m2.sum())
    out["rating"] = out["rating"].astype(object)
    m = pick(rng, n, .003); out.loc[m, "rating"] = rng.choice([0, 6, 10], size=m.sum()); log["review_rating_out_of_range"] = int(m.sum())
    m = pick(rng, n, .002); out.loc[m, "rating"] = out.loc[m, "rating"].apply(lambda r: f"{r} stars"); log["review_rating_as_text"] = int(m.sum())
    out["is_recommended"] = out["is_recommended"].astype(object)
    m = pick(rng, n, .12); out.loc[m, "is_recommended"] = np.nan; log["review_is_recommended_missing"] = int(m.sum())
    m = pick(rng, n, .01) & out.is_recommended.notna()
    out.loc[m, "is_recommended"] = out.loc[m, "is_recommended"].map({1: "Yes", 0: "No"}); log["review_recommended_as_text"] = int(m.sum())
    m = pick(rng, n, .01) & (out.rating.apply(lambda r: r == 1)); out.loc[m, "is_recommended"] = 1; log["review_rating_rec_conflict"] = int(m.sum())
    m = pick(rng, n, .003) & out.helpfulness.notna(); out.loc[m, "helpfulness"] = out.loc[m, "helpfulness"] * 100; log["review_helpfulness_percent_scale"] = int(m.sum())
    m = pick(rng, n, .01) & (out.total_feedback_count > 0); out.loc[m, "total_feedback_count"] += 3; log["review_feedback_count_inconsistent"] = int(m.sum())
    # dates in mixed formats + future dates
    def fmt_date(d):
        t = pd.Timestamp(d)
        return rng.choice([t.strftime("%m/%d/%Y"), t.strftime("%d-%b-%Y"), t.strftime("%Y-%m-%dT%H:%M:%S")])
    m = pick(rng, n, .15); out.loc[m, "submission_time"] = out.loc[m, "submission_time"].apply(fmt_date); log["review_date_other_format"] = int(m.sum())
    m = pick(rng, n, .001); out.loc[m, "submission_time"] = "2031-01-01"; log["review_date_future"] = int(m.sum())
    m = pick(rng, n, .001); out.loc[m, "submission_time"] = "not a date"; log["review_date_unparseable"] = int(m.sum())
    # text issues
    m = pick(rng, n, .03); out.loc[m, "review_text"] = out.loc[m, "review_text"].apply(lambda t: f"<p>{t}</p><br/>&amp; more soon!"); log["review_text_html"] = int(m.sum())
    m = pick(rng, n, .01); out.loc[m, "review_text"] += " Bought it at https://example-shop.com/deal"; log["review_text_url"] = int(m.sum())
    m = pick(rng, n, .002); out.loc[m, "review_text"] += " Contact me at jane.doe123@mail.com"; log["review_text_email_pii"] = int(m.sum())
    m = pick(rng, n, .02); out.loc[m, "review_text"] = "Itâ€™s my 2nd tube â€“ " + out.loc[m, "review_text"]; log["review_text_mojibake"] = int(m.sum())
    m = pick(rng, n, .05); out.loc[m, "review_text"] += " " + rng.choice(INCENTIVE, size=m.sum()); log["review_text_incentivised"] = int(m.sum())
    m = pick(rng, n, .03); out.loc[m, "review_text"] += rng.choice([" 😍😍", " 👎", " 🙌✨"], size=m.sum()); log["review_text_emoji"] = int(m.sum())
    m = pick(rng, n, .01); out.loc[m, "review_text"] = rng.choice(FOREIGN, size=m.sum()); log["review_text_non_english"] = int(m.sum())
    m = pick(rng, n, .01); out.loc[m, "review_text"] = rng.choice(["", "   ", "ok", ".", np.nan], size=m.sum()); log["review_text_empty_or_trivial"] = int(m.sum())
    m = pick(rng, n, .005); out.loc[m, "product_id"] = "P999" + pd.Series(range(m.sum())).astype(str).values; log["review_orphan_product_id"] = int(m.sum())
    # duplicates: exact + syndicated (same author & text on another product of same brand)
    exact = out.sample(frac=.015, random_state=2); log["review_exact_duplicates"] = len(exact)
    synd = out.sample(frac=.01, random_state=3).copy()
    synd["product_id"] = synd["product_id"].map(lambda p: f"P{400000 + int(rng.integers(0, 1200))}")
    log["review_syndicated_copies"] = len(synd)
    return pd.concat([out, exact, synd], ignore_index=True).sample(frac=1, random_state=4).reset_index(drop=True)


def main(n_products=1200, n_reviews=60000, seed=42):
    rng = np.random.default_rng(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    products = make_products(rng, n_products)
    reviews = make_reviews(rng, products, n_reviews)
    log = {"seed": seed, "n_products_clean": n_products, "n_reviews_clean": n_reviews}
    raw_products = inject_product_issues(rng, products, log)
    raw_reviews = inject_review_issues(rng, reviews, log)
    raw_products.to_csv(OUT / "product_info.csv", index=False)
    half = len(raw_reviews) // 2  # Sephora ships reviews in several files
    raw_reviews.iloc[:half].to_csv(OUT / "reviews_0-30000.csv")
    raw_reviews.iloc[half:].to_csv(OUT / "reviews_30000-end.csv")
    log["n_products_raw"] = len(raw_products)
    log["n_reviews_raw"] = len(raw_reviews)
    (OUT / "injection_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", type=int, default=1200)
    ap.add_argument("--reviews", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.products, a.reviews, a.seed)
