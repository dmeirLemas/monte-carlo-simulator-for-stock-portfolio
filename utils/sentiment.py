import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Tuple

import mwclient
import pandas as pd
from numpy import mean
from progress_bar import ProgressBar
from silencer import silencer


def get_sp500_companies() -> pd.Series:
    """
    Fetch the list of S&P 500 companies using yfinance.
    :return: List of company names.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table["Security"]


def fetch_revisions_for_company(site, company, prog_bar: ProgressBar, lock):
    page = site.pages[company]
    revisions = list(page.revisions())
    revisions.sort(key=lambda rev: rev["timestamp"])
    with lock:
        prog_bar.increment()
    return company, revisions


def fetch_edits(companies: pd.Series) -> Dict[str, List[Tuple]]:
    revisions_dict: Dict[str, List[Tuple]] = {}
    site = mwclient.Site("en.wikipedia.org")
    lock = threading.Lock()

    prog_bar = ProgressBar(
        total=len(companies), program_name="to fetch company revision data."
    )
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(fetch_revisions_for_company, site, company, prog_bar, lock)
            for company in companies
        ]

        for future in as_completed(futures):
            company, revisions = future.result()
            if len(revisions) > 0:
                revisions_dict[company] = [
                    (rev["timestamp"], rev["comment"])
                    for rev in revisions
                    if "comment" in rev
                ]

    return revisions_dict


def find_sentiment_batch(sentiment_pipeline, texts: List[str]):
    sentiments: List[Dict[Any, Any]] = sentiment_pipeline(texts)
    results = []
    for sentiment in sentiments:
        if sentiment is not None:
            score = sentiment["score"]
            if sentiment["label"] == "NEGATIVE":
                score *= -1
            results.append(score)
    return results


def format_info(
    para_dict: Dict[str, List[Tuple]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    formatted_para_dict = {}
    for company in para_dict.keys():
        formatted_para_dict[company] = defaultdict(dict)
        for timestamp, comment in para_dict[company]:
            date = time.strftime("%Y-%m-%d", timestamp)
            if date not in formatted_para_dict[company]:
                formatted_para_dict[company][date] = dict(comments=list(), edit_count=0)

            formatted_para_dict[company][date]["edit_count"] += 1
            formatted_para_dict[company][date]["comments"].append(comment)

    return formatted_para_dict


def analyze_sentiment(
    sentiment_pipeline, formatted_para_dict: Dict[str, Dict[str, Dict[str, Any]]]
):
    for company in formatted_para_dict:
        for _, data in formatted_para_dict[company].items():
            sentiments = find_sentiment_batch(sentiment_pipeline, data["comments"])
            data["sentiment"] = mean(sentiments)
            data["neg_sentiment"] = len([s for s in sentiments if s < 0]) / len(
                sentiments
            )
            del data["comments"]

    return formatted_para_dict


def save_results(
    results: Dict[str, Dict[str, Dict[str, Any]]], rolling_days: int
) -> None:
    for company in results:
        df = pd.DataFrame.from_dict(results[company], orient="index")
        df.index = pd.to_datetime(df.index)

        date_range = pd.date_range(start=df.index[0], end=datetime.today())

        df = df.reindex(date_range, fill_value=0)

        df = df.rolling(rolling_days).mean().dropna()

        if "Sentiment_Scores" not in os.listdir():
            os.mkdir("Sentiment_Scores")

        df.to_csv(f"Sentiment_Scores/{company}.csv", index_label="Date")


def get_sentiments(companies: pd.Series):
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    MODEL_REVISION = "af0f99b"

    with silencer():
        from transformers import pipeline
        from transformers.utils import logging

        logging.set_verbosity_error()
        sentiment_pipeline = pipeline(model=MODEL_NAME, revision=MODEL_REVISION)

    para_dict_wiki = format_info(fetch_edits(companies))

    results = analyze_sentiment(sentiment_pipeline, para_dict_wiki)

    save_results(results, 365)


if __name__ == "__main__":
    companies = get_sp500_companies()[:1]
    get_sentiments(companies)
