# Feature catalogue (Week 4)

| Feature | Family | Definition | Why |
|---|---|---|---|
| profile_completeness | Customer behaviour | Number of the 4 optional profile fields the reviewer filled (0-4) | Engaged shoppers disclose more; engagement may relate to satisfaction |
| reviewer_prior_reviews | Customer behaviour | Reviews the same reviewer wrote before this one | Experienced reviewers are more critical / more trusted |
| reviewer_prior_mean_rating | Customer behaviour | Mean rating the reviewer gave in earlier reviews (NaN if first) | Captures 'harsh' vs 'generous' reviewers |
| is_incentivised | Customer behaviour | Review mentions a free / gifted product | Incentivised reviews skew positive |
| month_sin / month_cos | Market trend | Cyclical encoding of review month | Seasonality (e.g. sunscreen in summer, gifting in Nov-Dec) without a fake Dec->Jan jump |
| is_holiday_season | Market trend | Review written in November or December | Gift purchases and sale events |
| days_since_launch, is_early_adopter | Market trend | Days since product's first review; first 90 days flag | Early buyers are enthusiasts; ratings often drift down after launch hype |
| product_prior_bayes_rating | Product momentum | Bayesian mean of the product's ratings BEFORE this review (prior m=20) | Leak-free replacement for Week 3's product_bayes_rating, which used future reviews |
| product_prior_rec_rate | Product momentum | Smoothed recommendation rate before this review | Social proof the shopper saw at purchase time |
| product_momentum | Product momentum | Mean of last 10 prior ratings minus long-run prior mean | Detects products trending up (positive) or down (e.g. reformulation, batch issues) |
| review_velocity | Product momentum | Reviews per month up to this review | Proxy for sales velocity / buzz |
| price_rel_category | Price positioning | Price per mL / category median price per mL | A $40 serum and $40 cleanser are not equally 'expensive'; relative price is comparable across categories |
| price_tier | Price positioning | Quartile of price per mL within category | Non-linear price effects; easy to explain |
| strong_actives | Formulation | Count of potentially irritating actives (retinol, salicylic, lactic, ascorbic acid) | Powerful actives drive results but also irritation complaints |
| fragrance_free_verified | Formulation | Claims fragrance-free AND no fragrance in INCI | Separates genuine from mislabelled claims |
| dry_x_fragrance | Interaction | Dry-skin reviewer AND product contains fragrance | Directly tests H4 |
| skin_mismatch | Interaction | Product targets the opposite skin type (dry vs oily) | Strongest expected dissatisfaction case |
| oily_x_rich_texture | Interaction | Oily-skin reviewer AND night cream / face oil | Texture-skin conflict |
| premium_x_first_purchase | Interaction | Premium/luxury product AND reviewer's first review | High expectations of first-time premium buyers |
| aspect_* (6) | Text | Counts of aspect keywords: skin reaction, texture, scent, value, packaging, efficacy | Explainable first version of the aspect analysis in H3 (replaced by a transformer model later) |
| exclamations, has_emoji, caps_ratio | Text | Emotional intensity markers | Strong emotion -> extreme ratings |
| brand_te | Encoding | Out-of-fold (5-fold) smoothed recommendation rate of the brand (m=50) | Captures brand reputation in one column instead of 30 one-hot columns, without target leakage |
| brand_freq | Encoding | Share of reviews belonging to the brand | Brand size / visibility |