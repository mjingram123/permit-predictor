import glob
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_DIR = os.path.join(ROOT, "cache", "parcel_history")
BATCH_DIR = os.path.join(WORK_DIR, "batches")
SEED_CACHE = os.path.join(WORK_DIR, "seed_parcel_permits.csv")
INPUT_FILES = [
    os.path.join(ROOT, "data", "permits_2010_2019.csv.gz"),
    os.path.join(ROOT, "data", "permits_2020_present.csv.gz"),
]
OUTPUT_PATH = os.path.join(ROOT, "data", "parcel_history.csv")

DATASETS = ["b6ii-mhed", "n3xg-rixm", "gwh9-jnip"]
SELECT_COLUMNS = "permit_nbr,apn,permit_type,submitted_date,issue_date"
BATCH_SIZE = 100
DEMOLITION_TYPES = {"Bldg-Demolition", "Nonbldg-Demolition"}
THREE_YEARS = pd.Timedelta(days=1095)
FIVE_YEARS = pd.Timedelta(days=1826)
SIBLING_FLOOR = pd.Timedelta(days=31)
FLAG_COLUMNS = ["prior_demo_3y", "prior_grading_3y", "prior_newbldg_5y", "prior_newbldg_issued", "has_apn"]
OUTPUT_COLUMNS = [
    "permit_nbr", "prior_demo_3y", "prior_grading_3y", "prior_newbldg_5y",
    "prior_newbldg_issued", "n_prior_permits_5y", "days_since_last_permit", "has_apn",
]


def load_applications():
    frames = [pd.read_csv(path, dtype=str, usecols=["permit_nbr", "apn", "submitted_date"]) for path in INPUT_FILES]
    applications = pd.concat(frames, ignore_index=True)
    applications["submitted"] = pd.to_datetime(applications["submitted_date"], errors="coerce")
    applications["apn"] = applications["apn"].str.strip()
    applications.loc[applications["apn"] == "", "apn"] = np.nan
    return applications


def seed_cache():
    os.makedirs(BATCH_DIR, exist_ok=True)


def seeded_parcels():
    if not os.path.exists(SEED_CACHE):
        return set()
    return set(pd.read_csv(SEED_CACHE, dtype=str, usecols=["apn"])["apn"].dropna())


def download_batch(dataset, parcels, destination):
    where = "apn in (" + ",".join("'" + parcel + "'" for parcel in parcels) + ")"
    query = urllib.parse.urlencode(
        {"$select": SELECT_COLUMNS, "$where": where, "$limit": "50000"},
        quote_via=urllib.parse.quote,
    )
    url = f"https://data.lacity.org/resource/{dataset}.csv?{query}"
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "student-research-script"})
            payload = urllib.request.urlopen(request, timeout=120).read()
            with open(destination, "wb") as handle:
                handle.write(payload)
            return
        except Exception as error:
            print("retry", dataset, destination, error, flush=True)
            time.sleep(5 * (attempt + 1))
    sys.exit("download failed: " + destination)


def fetch_missing_parcels(parcels):
    missing = sorted(set(parcels) - seeded_parcels())
    print("parcels to fetch:", len(missing), flush=True)
    for dataset in DATASETS:
        for start in range(0, len(missing), BATCH_SIZE):
            destination = os.path.join(BATCH_DIR, f"{dataset}_{start // BATCH_SIZE:04d}.csv")
            if os.path.exists(destination):
                continue
            download_batch(dataset, missing[start:start + BATCH_SIZE], destination)
            time.sleep(1)
    return len(missing)


def load_parcel_permits():
    paths = [SEED_CACHE] + sorted(glob.glob(os.path.join(BATCH_DIR, "*.csv")))
    frames = []
    for path in paths:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        frames.append(pd.read_csv(path, dtype=str, usecols=["permit_nbr", "apn", "permit_type", "submitted_date", "issue_date"]))
    permits = pd.concat(frames, ignore_index=True).drop_duplicates("permit_nbr")
    permits = permits[permits["apn"].notna()].copy()
    permits["submitted"] = pd.to_datetime(permits["submitted_date"], errors="coerce")
    permits["issued"] = pd.to_datetime(permits["issue_date"], errors="coerce")
    permits = permits[permits["submitted"].notna()]
    permits["is_demolition"] = permits["permit_type"].isin(DEMOLITION_TYPES)
    permits["is_grading"] = permits["permit_type"].eq("Grading")
    middle_segment = permits["permit_nbr"].str.split("-").str[1].fillna("")
    permits["is_new_building"] = permits["permit_type"].eq("Bldg-New") & middle_segment.str.endswith("0000")
    return permits


def features_for_application(parcel_permits, applied):
    prior = parcel_permits[parcel_permits["submitted"] < applied]
    if prior.empty:
        return dict(prior_demo_3y=0, prior_grading_3y=0, prior_newbldg_5y=0, prior_newbldg_issued=0,
                    n_prior_permits_5y=0, days_since_last_permit=np.nan)
    within_three = prior["submitted"] >= applied - THREE_YEARS
    within_five = prior["submitted"] >= applied - FIVE_YEARS
    earlier_new = prior[prior["is_new_building"] & within_five & (prior["submitted"] <= applied - SIBLING_FLOOR)]
    return dict(
        prior_demo_3y=int((prior["is_demolition"] & within_three).any()),
        prior_grading_3y=int((prior["is_grading"] & within_three).any()),
        prior_newbldg_5y=int(len(earlier_new) > 0),
        prior_newbldg_issued=int((earlier_new["issued"] < applied).any()),
        n_prior_permits_5y=int(within_five.sum()),
        days_since_last_permit=(applied - prior["submitted"].max()).days,
    )


def build_history(applications, permits):
    by_parcel = {parcel: group for parcel, group in permits.groupby("apn")}
    rows = []
    for permit_nbr, parcel, applied in zip(applications["permit_nbr"], applications["apn"], applications["submitted"]):
        row = dict(permit_nbr=permit_nbr, has_apn=int(pd.notna(parcel)))
        empty = dict(prior_demo_3y=0, prior_grading_3y=0, prior_newbldg_5y=0, prior_newbldg_issued=0,
                     n_prior_permits_5y=0, days_since_last_permit=np.nan)
        if pd.isna(parcel) or pd.isna(applied) or parcel not in by_parcel:
            row.update(empty)
        else:
            row.update(features_for_application(by_parcel[parcel], applied))
        rows.append(row)
    history = pd.DataFrame(rows)[OUTPUT_COLUMNS]
    history["days_since_last_permit"] = history["days_since_last_permit"].astype("Int64")
    return history


def self_check(history, applications):
    expected = set(applications["permit_nbr"])
    assert len(history) == len(expected), "row count differs from inputs"
    assert not history["permit_nbr"].duplicated().any(), "duplicate permit_nbr"
    assert set(history["permit_nbr"]) == expected, "permit_nbr set differs from inputs"
    for column in FLAG_COLUMNS:
        assert history[column].isin([0, 1]).all(), column + " is not 0/1"
    assert (history.loc[history["has_apn"] == 0, OUTPUT_COLUMNS[1:7]].fillna(0) == 0).all().all(), "no-apn rows must be zero"
    print("rows:", len(history))
    for column in FLAG_COLUMNS:
        print(f"share {column} = 1: {history[column].mean():.4f}")
    print("share with days_since_last_permit blank:", round(history["days_since_last_permit"].isna().mean(), 4))
    print("self-check passed")


def main():
    applications = load_applications()
    seed_cache()
    parcel_count = fetch_missing_parcels(applications["apn"].dropna().unique())
    permits = load_parcel_permits()
    history = build_history(applications, permits)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    history.to_csv(OUTPUT_PATH, index=False)
    print("parcels fetched this design:", parcel_count)
    self_check(history, applications)


if __name__ == "__main__":
    main()
