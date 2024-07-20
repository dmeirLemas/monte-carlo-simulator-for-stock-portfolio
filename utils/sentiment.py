import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List

import pandas as pd
import mwclient
from numpy import mean
from transformers import pipeline


sentiment_pipeline = pipeline("sentiment-analysis")


def get_sp500_companies() -> Iterable[str]:
    """
    Fetch the list of S&P 500 companies using yfinance.
    :return: List of company names.
    """
    # Get the list of S&P 500 companies
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]

    return table["Security"]


def fetch_edits(companies: Iterable[str]) -> Dict[str, List[dict]]:
    revisions_dict: Dict[str, List[dict]] = {}
    site = mwclient.Site("en.wikipedia.org")
    for company in companies:
        print(f"Fetching {company}")
        page = site.pages[company]

        revisions = list(page.revisions())
        revisions.sort(key=lambda rev: rev["timestamp"])

        revisions_dict[company] = revisions

    return revisions_dict


def find_sentiment(text: str):
    global sentiment_pipeline
    sentiment = sentiment_pipeline([text[:250]])[0]
    score = sentiment["score"]
    if sentiment["label"] == "NEGATIVE":
        score *= -1

    return score


def main():
    companies = get_sp500_companies()[:1]
    revision_dict = fetch_edits(companies)

    edits = {}
    for company in companies:
        edits[company] = defaultdict(dict)
        for rev in revision_dict[company]:
            date = time.strftime("%Y-%m-%d", rev["timestamp"])
            if date not in edits[company]:
                edits[company][date] = dict(sentiments=list(), edit_count=0)

            edits[company][date]["edit_count"] += 1

            comment = rev["comment"]

            edits[company][date]["sentiments"].append(find_sentiment(comment))

    for company in companies:
        for key in edits[company]:
            if len(edits[company][key]["sentiments"]) > 0:
                edits[company][key]["sentiment"] = mean(
                    edits[company][key]["sentiments"]
                )
                edits[company][key]["neg_sentiment"] = len(
                    [s for s in edits[company][key]["sentiments"] if s < 0]
                ) / len(edits[company][key]["sentiments"])

            else:
                edits[company][key]["sentiment"] = 0
                edits[company][key]["neg_sentiment"] = 0

            del edits[company][key]["sentiments"]

    for company in edits:
        edits_df = pd.DataFrame.from_dict(edits[company], orient="index")
        edits_df.index = pd.to_datetime(edits_df.index)

        date_range = pd.date_range(start=edits_df.index[0], end=datetime.today())

        edits_df = edits_df.reindex(date_range, fill_value=0)

        edits_df = edits_df.rolling(365).mean().dropna()

        edits_df.to_csv(f"{company}.csv", index_label="Date")


if __name__ == "__main__":
    main()
