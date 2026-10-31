# Decoding the Skincare Shopper
**Data Science with Python Internship — YuvaIntern (Henry Harvin)**
Author: Aayush Gautam

Predicting skincare product recommendation and satisfaction from customer reviews, product attributes (price, ingredients, claims) and reviewer skin profiles, in the Beauty & Wellness industry.

## Problem
Online beauty shoppers rely on reviews, but a single average star rating hides how a product performs for different skin types. This project finds what drives a customer to recommend a skincare product, predicts recommendation, and prototypes skin-type-aware product suggestions.

## Data
Primary: **Sephora Products & Skincare Reviews** (Kaggle). Validation: **Amazon Reviews 2023 (Beauty)**. Enrichment & context: Open Beauty Facts, EU CosIng, FDA CAERS, California Chemicals in Cosmetics, Google Trends.
Full list with links and licences: [data/DATA_REGISTER.md](data/DATA_REGISTER.md). Raw data is not committed.

## Quick start
```bash
pip install -r requirements.txt

# Week 3 - cleaning pipeline (simulated raw data with known, injected issues)
python src/simulate_raw_data.py --seed 42
python -m src.cleaning.pipeline                 # writes data/processed + reports/
python -m unittest discover -s tests -t . -v    # 9 unit tests
python -m src.reporting.make_figures            # report figures

# Real data
python src/download_data.py sephora             # needs a Kaggle API token
python -m src.cleaning.pipeline --raw data/raw/sephora
```

## Project structure
```
beauty-wellness/
├── configs/
│   └── cleaning.yaml            # every threshold, mapping and path used by the pipeline
├── data/
│   ├── DATA_REGISTER.md         # all sources, links, licences (Week 2)
│   ├── raw/                     # untouched downloads / simulated raw data   (not tracked)
│   ├── interim/                 # merged, typed tables                       (not tracked)
│   └── processed/               # clean, model-ready tables                  (not tracked)
├── docs/
│   ├── week1/                   # problem definition report
│   ├── week2/                   # data sourcing strategy report
│   └── week3/                   # cleaning pipeline report, flowchart (.dot), figures/
├── notebooks/                   # exploratory notebooks (Week 4+)
├── reports/                     # generated: cleaning log, before/after data quality
├── src/
│   ├── download_data.py         # Week 2: fetch Sephora, FDA CAERS, Google Trends
│   ├── data_quality.py          # Week 2: first quality profile of real Sephora data
│   ├── simulate_raw_data.py     # Week 3: raw data generator with injected issues
│   ├── cleaning/                # Week 3: 9-stage pipeline
│   │   ├── steps.py             #   schema, standardise, validity, duplicates, text, INCI
│   │   ├── impute_outliers.py   #   missing values, outliers, feature engineering
│   │   ├── preprocess.py        #   scikit-learn scaling/encoding, time-based split
│   │   ├── log.py               #   audit log
│   │   └── pipeline.py          #   runs every stage, writes outputs
│   └── reporting/
│       └── make_figures.py      # report figures
├── tests/
│   └── test_cleaning.py         # 9 unit tests
├── requirements.txt
└── README.md
```

## Progress
| Week | Task | Status |
|---|---|---|
| 1 | Orientation & problem definition — [report](docs/week1/) | ✅ Done |
| 2 | Data identification, sourcing strategy & analysis plan — [report](docs/week2/), [data register](data/DATA_REGISTER.md), [scripts](src/) | ✅ Done |
| 3 | Data cleaning & pre-processing pipeline — [report](docs/week3/), [code](src/cleaning/), [cleaning log](reports/cleaning_log.md), [before/after](reports/data_quality_before_after.md) | ✅ Done |
| 4 | Exploratory analysis & hypothesis testing | ⏳ |
| 5 | Predictive modelling & SHAP | ⏳ |
| 6 | Recommender prototype & dashboard | ⏳ |

## Tools
Python, pandas, NumPy, Matplotlib, Seaborn, Plotly, SciPy, statsmodels, scikit-learn, XGBoost/LightGBM, NLTK/spaCy, VADER, Hugging Face Transformers, BERTopic, SHAP, Streamlit
