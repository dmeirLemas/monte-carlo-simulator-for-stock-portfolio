import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Iterable, List

import mwclient
import pandas as pd
from numpy import mean
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")


def get_sp500_companies() -> Iterable[str]:
    """
    Fetch the list of S&P 500 companies using yfinance.
    :return: List of company names.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table["Security"]


def fetch_revisions_for_company(site, company):
    print(f"Fetching {company}")
    page = site.pages[company]
    revisions = list(page.revisions())
    revisions.sort(key=lambda rev: rev["timestamp"])
    return company, revisions


def fetch_edits(companies: Iterable[str]) -> Dict[str, List[dict]]:
    revisions_dict: Dict[str, List[dict]] = {}
    site = mwclient.Site("en.wikipedia.org")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(fetch_revisions_for_company, site, company)
            for company in companies
        ]

        for future in as_completed(futures):
            company, revisions = future.result()
            revisions_dict[company] = revisions

    return revisions_dict


def find_sentiment_batch(texts: List[str]):
    sentiments: List[Dict[Any, Any]] = sentiment_pipeline(texts)
    results = []
    for sentiment in sentiments:
        score = sentiment["score"]
        if sentiment["label"] == "NEGATIVE":
            score *= -1
        results.append(score)
    return results


def get_sentiment(companies: List[str]):
    revision_dict = fetch_edits(companies)

    edits = {}
    for company in companies:
        edits[company] = defaultdict(dict)
        for rev in revision_dict[company]:
            date = time.strftime("%Y-%m-%d", rev["timestamp"])
            if date not in edits[company]:
                edits[company][date] = dict(comments=list(), edit_count=0)

            edits[company][date]["edit_count"] += 1
            comment = rev["comment"]
            edits[company][date]["comments"].append(comment)

    for company in companies:
        for date, data in edits[company].items():
            sentiments = find_sentiment_batch(data["comments"])
            data["sentiment"] = mean(sentiments)
            data["neg_sentiment"] = len([s for s in sentiments if s < 0]) / len(
                sentiments
            )

            del data["comments"]

    for company in companies:
        edits_df = pd.DataFrame.from_dict(edits[company], orient="index")
        edits_df.index = pd.to_datetime(edits_df.index)

        date_range = pd.date_range(start=edits_df.index[0], end=datetime.today())

        edits_df = edits_df.reindex(date_range, fill_value=0)

        edits_df = edits_df.rolling(365).mean().dropna()

        if "Sentiment_Scores" not in os.listdir():
            os.mkdir("Sentiment_Scores")

        edits_df.to_csv(f"Sentiment_Scores/{company}.csv", index_label="Date")
