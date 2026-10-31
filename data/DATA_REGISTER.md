# Data Register

Raw data is **not** stored in this repository (size and licence reasons). Use `src/download_data.py` to re-create it. Each download is logged automatically in `data/download_log.csv`.

| ID | Source | Link | Licence / terms | Role | Status |
|---|---|---|---|---|---|
| S1 | Sephora Products & Skincare Reviews (Kaggle) | https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews | See Kaggle licence tab | **Primary** | Planned (Week 2–3) |
| S2 | Amazon Reviews 2023 – All_Beauty / Beauty_and_Personal_Care | https://amazon-reviews-2023.github.io/ | Academic use; cite Hou et al. 2024 | Validation | Planned |
| S3 | Skincare ABSA dataset (Bangladeshi e-commerce) | https://data.mendeley.com/datasets/6vmrcypy7c | CC BY 4.0 | NLP validation | Planned |
| S4 | Beauty Product Reviews RecSys dataset | https://data.mendeley.com/datasets/n9sk45scmt | See Mendeley page | Optional | Candidate |
| S5 | Open Beauty Facts | https://world.openbeautyfacts.org/data | ODbL / DbCL | Enrichment | Planned |
| S6 | EU CosIng ingredient database | https://ec.europa.eu/growth/tools-databases/cosing/ | European Commission | Enrichment (ingredient functions) | Planned |
| S7 | California Chemicals in Cosmetics (CSCP) | https://catalog.data.gov/dataset/chemicals-in-cosmetics-d55bf | CC BY | Context | Planned |
| S8 | FDA CAERS – cosmetics adverse events | https://open.fda.gov/data/caers | Public domain (CC0) | Context | Planned |
| S9 | Google Trends | https://trends.google.com | Google ToS | Context (India vs world) | Planned |
| S10 | Reddit Data API | https://www.reddit.com/dev/api | Reddit Data API terms | Stretch | Optional |
| S11 | YouTube Data API v3 | https://developers.google.com/youtube/v3 | Google API ToS | Stretch | Optional |
| S12 | Makeup API | http://makeup-api.herokuapp.com | Open | – | Rejected (makeup only) |
| S13 | Kaggle Cosmetics dataset | https://www.kaggle.com/datasets/kingabzpro/cosmetics-datasets | See Kaggle | Cross-check | Candidate |

## Folder layout

```
data/raw/        untouched downloads
data/interim/    merged and typed tables
data/processed/  analysis-ready tables
data/external/   enrichment tables (CosIng, Trends, CAERS ...)
```
