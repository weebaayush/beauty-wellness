"""Week 2 - download the project's public data sources.

Usage:
    python src/download_data.py sephora      # needs Kaggle API token (~/.kaggle/kaggle.json)
    python src/download_data.py caers        # FDA cosmetics adverse events (openFDA)
    python src/download_data.py trends       # Google Trends, India vs worldwide
    python src/download_data.py all

Raw files go to data/raw or data/external and are NOT committed to GitHub
(see .gitignore). Every download is logged in data/download_log.csv.
"""
import csv
import datetime as dt
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXT = ROOT / "data" / "external"
LOG = ROOT / "data" / "download_log.csv"


def log(source, path, url):
    path = Path(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else ""
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "source", "file", "url", "sha256_16"])
        w.writerow([dt.date.today(), source, path.relative_to(ROOT), url, sha])


def sephora():
    """S1 - Sephora Products and Skincare Reviews (Kaggle)."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    out = RAW / "sephora"
    out.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("nadyinky/sephora-products-and-skincare-reviews", path=out, unzip=True)
    for f in out.glob("*.csv"):
        log("S1 Sephora", f, "https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews")
    print("Sephora files:", sorted(p.name for p in out.glob("*.csv")))


def caers(limit=1000, max_records=25000):
    """S8 - FDA CAERS reports where the industry is cosmetics (openFDA food/event API)."""
    import json
    import requests

    EXT.mkdir(parents=True, exist_ok=True)
    url = "https://api.fda.gov/food/event.json"
    params = {"search": 'products.industry_name:"Cosmetics"', "limit": limit, "skip": 0}
    rows = []
    while params["skip"] < max_records:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 404:  # no more results
            break
        r.raise_for_status()
        batch = r.json().get("results", [])
        if not batch:
            break
        rows.extend(batch)
        params["skip"] += limit
        time.sleep(0.3)  # stay well under the public rate limit
    path = EXT / "caers_cosmetics.json"
    path.write_text(json.dumps(rows))
    log("S8 FDA CAERS", path, url)
    print(f"CAERS cosmetics reports saved: {len(rows)}")


TREND_TERMS = ["niacinamide", "hyaluronic acid", "retinol", "vitamin c serum", "sunscreen",
               "ceramide", "salicylic acid", "fragrance free moisturizer", "acne", "oily skin"]


def trends():
    """S9 - Google Trends monthly interest, India vs worldwide (unofficial pytrends)."""
    import pandas as pd
    from pytrends.request import TrendReq

    EXT.mkdir(parents=True, exist_ok=True)
    py = TrendReq(hl="en-US", tz=330)
    frames = []
    for geo in ["IN", ""]:
        for i in range(0, len(TREND_TERMS), 5):  # pytrends accepts max 5 terms per request
            terms = TREND_TERMS[i:i + 5]
            py.build_payload(terms, timeframe="2019-01-01 2026-10-01", geo=geo)
            df = py.interest_over_time().drop(columns="isPartial", errors="ignore")
            df = df.reset_index().melt(id_vars="date", var_name="term", value_name="interest")
            df["geo"] = geo or "WORLD"
            frames.append(df)
            time.sleep(5)
    path = EXT / "google_trends.csv"
    pd.concat(frames).to_csv(path, index=False)
    log("S9 Google Trends", path, "https://trends.google.com")
    print("Google Trends saved:", path)


if __name__ == "__main__":
    jobs = {"sephora": sephora, "caers": caers, "trends": trends}
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in jobs.items():
        if choice in (name, "all"):
            print(f"--- {name} ---")
            try:
                fn()
            except Exception as e:  # keep going so one failing source doesn't stop the rest
                print(f"[{name}] failed: {e}")
