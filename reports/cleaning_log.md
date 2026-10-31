# Cleaning log

| Step | Action | Rows affected |
|---|---|---|
| schema | products: all 8 required columns present | 0 |
| schema | reviews: all 12 required columns present | 0 |
| standardise | price strings parsed to float ($, 'USD', spaces removed) | 58 |
| standardise | brand names unified via canonical key | 67 |
| standardise | category spellings mapped to 6 canonical categories | 110 |
| standardise | size text parsed to millilitres (oz x 29.5735) | 1,002 |
| standardise | highlights parsed from text to lists | 921 |
| standardise | rating text ('5 stars') converted to numbers | 117 |
| standardise | is_recommended Yes/No/1.0 values unified to 1/0 | 527 |
| standardise | skin_type variants mapped to 4 canonical labels | 4,337 |
| standardise | skin_tone variants mapped to 13 canonical labels | 2,587 |
| standardise | dates in 4 formats parsed to one datetime type | 9,231 |
| validity | prices outside $1-$1000 set to missing | 10 |
| validity | negative loves_count corrected (sign error -> absolute value) | 14 |
| validity | ratings outside 1-5 set to missing | 166 |
| validity | helpfulness on 0-100 scale rescaled to 0-1 | 78 |
| validity | total_feedback_count recomputed as pos + neg where inconsistent | 282 |
| validity | dates after snapshot (2023-03-31) or before 2008-01-01 set to missing | 70 |
| validity | unparseable dates left as missing | 63 |
| validity | 1-star reviews marked 'recommended' flagged as conflicts | 3 |
| duplicates | duplicate product rows removed | 12 |
| duplicates | exact duplicate reviews removed (same author, product, text) | 902 |
| duplicates | syndicated copies removed (same author + text on another product) | 553 |
| integrity | reviews whose product_id has no product record removed | 302 |
| text | HTML tags/entities cleaned | 1,805 |
| text | URLs cleaned | 576 |
| text | e-mail addresses (PII) cleaned | 123 |
| text | mojibake characters cleaned | 1,164 |
| text | reviews with < 3 words flagged (excluded from NLP only) | 637 |
| text | possibly non-English reviews flagged | 585 |
| text | incentivised reviews flagged (free product / Influenster) | 2,958 |
| text | author_id replaced by salted SHA-256 hash (privacy) | 59,743 |
| ingredients | legal disclaimer text removed | 580 |
| ingredients | marketing descriptions ('-Niacinamide: ...') removed | 398 |
| ingredients | ingredient synonyms unified (e.g. Aqua -> water, Parfum -> fragrance) | 1,135 |
| missing | missing product price recovered from review-level price snapshot | 35 |
| missing | remaining missing price imputed with brand x category median (groups >= 5) | 0 |
| missing | remaining missing price imputed with category median (groups >= 5) | 0 |
| missing | size left missing and flagged (not imputable without guessing) | 207 |
| missing | ingredients missing -> indicator column added | 73 |
| missing | highlights missing -> indicator column added | 289 |
| missing | skin_type missing -> 'unknown' category + indicator | 17,911 |
| missing | skin_tone missing -> 'unknown' category + indicator | 15,730 |
| missing | eye_color missing -> 'unknown' category + indicator | 20,906 |
| missing | hair_color missing -> 'unknown' category + indicator | 17,902 |
| missing | helpfulness missing (0 votes) -> Laplace-smoothed helpfulness score | 31,733 |
| missing | is_recommended missing -> kept for EDA, excluded from model training | 7,192 |
| missing | rating missing (after validity checks) -> kept, excluded from rating analyses | 161 |
| outliers | price-per-ml outliers flagged (|robust z| > 3.5 within category, log scale) | 4 |
| outliers | loves_count outliers by IQR rule on log scale (flagged, kept: viral products are real) | 9 |
| outliers | price_per_ml winsorised at 1th/99th percentiles (model copy) | 20 |
| outliers | loves_count winsorised at 1th/99th percentiles (model copy) | 24 |
| outliers | review word count capped at 99th percentile (model copy) | 578 |
| features | products claiming 'Fragrance Free' but listing fragrance in INCI flagged | 8 |
| features | 10 active-ingredient flags + fragrance flag + 6 claim flags created | 1,200 |
| features | skin_match feature (reviewer skin type == product 'Good for' claim) | 24,734 |
| features | Bayesian-adjusted product rating (prior m = 20 reviews, global mean C = 4.07) | 1,175 |
| features | time features: review_year, months_since_first_review | 59,614 |
| validation | final assertions passed (ranges, integrity, duplicates, PII) | 59,743 |
| normalise | model matrix: 90 features; train 41,952 / test 10,488 (time-based split) | 52,440 |