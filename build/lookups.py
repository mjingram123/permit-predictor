import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache" / "lookups"
USER_AGENT = "Mozilla/5.0"
PAUSE = 1.0

ARCGIS = "https://services1.arcgis.com/tzwalEyxl2rpamKs/arcgis/rest/services"
COMPLETED_LAYERS = [
    ("BiWeekly_Cases_Archived", 0),
    ("BiWeekly_Cases_Archived_2021", 0),
    ("BiWeekly_Cases_Archived_2022", 0),
    ("BiWeekly_Cases_Achived_2023", 0),
    ("CasesCompleted_2024", 0),
    ("BiWeekly_Cases_Archived_2025", 1),
    ("BiWeeklyCasesCompleted", 0),
]

LADBS_DATASETS = ["n3xg-rixm", "gwh9-jnip"]
LADBS_WHERE = "permit_type='Bldg-New' AND permit_sub_type in ('Commercial','Apartment')"
SEATTLE_DATASET = "76t5-zqzr"
SEATTLE_WHERE = (
    "permittypedesc='New' AND permittypemapped='Building' "
    "AND permitclass in ('Commercial','Industrial','Institutional','Multifamily')"
)

MIN_CASES = 30
PREFIX_TYPES = [
    ("CPC", "City Planning Commission decision"),
    ("DIR", "Director of Planning decision"),
    ("ZA", "Zoning Administrator decision"),
    ("VTT/TT", "Tract map (vesting tentative or tentative tract)"),
]
SUFFIX_TYPES = [
    ("TOC", "Transit-oriented incentive (TOC)", {"TOC"}),
    ("DB", "Density bonus", {"DB"}),
    ("SPR", "Site plan review", {"SPR"}),
    ("CU/CUB", "Conditional use permit (including alcohol, CUB)", {"CU", "CUB"}),
    ("ZC", "Zone change", {"ZC"}),
    ("GPA", "General plan amendment", {"GPA"}),
    ("EIR", "Environmental impact report", {"EIR"}),
]

CONSTRUCTION_START = pd.Timestamp("2012-01-01")
CONSTRUCTION_END = pd.Timestamp("2020-12-31")
MIN_COFO_GAP_DAYS = 30
DAYS_PER_MONTH = 365.25 / 12

JUNK_WORDS = [
    "carport", "patio cover", "canopy", "temporary structure", "private garage", "garage - private",
    "accessory", "single family", "duplex", "foundation only", "fence", "wall", "sign", "pool",
    "tent", "tank", "antenna", "tower", "miscellaneous", "shed", "trash", "retaining", "gazebo",
    "trellis", "equipment", "recreation room",
]
JUNK_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in JUNK_WORDS) + ")", re.I)
MULTI_RE = re.compile(r"apartment|dwelling - multiple|condo", re.I)
MIXED_WORK_RE = re.compile(r"mixed[\s-]*use|retail|commercial", re.I)
HOTEL_RE = re.compile(r"hotel|motel", re.I)
OFFICE_RE = re.compile(r"office|bank|clinic", re.I)
INDUSTRIAL_RE = re.compile(r"warehouse|manufacturing|storage building|industrial", re.I)
RETAIL_RE = re.compile(r"retail|grocery|restaurant|cafe|car wash|gas station|auto|store|market", re.I)
ASSET_ORDER = ["multifamily", "mixed-use", "office", "retail", "industrial", "hotel", "other commercial"]

PLAN_CHECK_ONLY_RE = re.compile(r"ZONING ONLY|PDPP|HSAP|PC ONLY|PARALLEL DESIGN|SB 330", re.I)
BENCH_YEARS = list(range(2012, 2024))

OUTPUTS = {
    "entitlement_lookup.csv": ["code", "plain_name", "n_cases", "p25_days", "median_days", "p75_days", "p90_days"],
    "construction_lookup.csv": ["asset_class", "n_permits", "p25_months", "median_months", "p75_months", "share_with_cofo"],
    "city_benchmark.csv": ["city", "filing_year", "n_applications", "share_12m", "share_24m", "median_days"],
}


def fetch(url, timeout=180):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def socrata_csv(domain, dataset, where, cache_name):
    path = CACHE / cache_name
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path, dtype=str)
    frames = []
    offset = 0
    page = 50000
    while True:
        query = urllib.parse.urlencode(
            {"$where": where, "$limit": page, "$offset": offset, "$order": ":id"},
            quote_via=urllib.parse.quote,
        )
        raw = fetch(f"https://{domain}/resource/{dataset}.csv?{query}")
        frame = pd.read_csv(io.BytesIO(raw), dtype=str)
        frames.append(frame)
        if len(frame) < page:
            break
        offset += page
        time.sleep(PAUSE)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(path, index=False)
    time.sleep(PAUSE)
    return out


def arcgis_layer(service, layer):
    path = CACHE / f"arcgis_{service}_{layer}.json"
    if path.exists() and path.stat().st_size > 2:
        return json.loads(path.read_text())
    rows = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "false",
                "resultOffset": offset,
                "resultRecordCount": 2000,
                "orderByFields": "OBJECTID",
                "f": "json",
            }
        )
        payload = json.loads(fetch(f"{ARCGIS}/{service}/FeatureServer/{layer}/query?{query}", timeout=90))
        features = payload.get("features", [])
        rows += [f["attributes"] for f in features]
        if len(features) < 2000 and not payload.get("exceededTransferLimit"):
            break
        offset += 2000
        time.sleep(PAUSE)
    path.write_text(json.dumps(rows))
    time.sleep(PAUSE)
    return rows


def pct(series, q):
    return float(np.percentile(series.to_numpy(dtype=float), q))


def parse_case_date(values):
    first = pd.to_datetime(values, format="%d-%b-%y", errors="coerce")
    second = pd.to_datetime(values, format="%m/%d/%Y", errors="coerce")
    return first.fillna(second)


def case_tokens(case, request_type):
    parts = case.upper().split("-")
    found = {p.strip() for p in parts[3:] if p.strip()}
    for item in str(request_type or "").split(";"):
        code = item.split("-")[0].strip().upper()
        if code and code != "NAN":
            found.add(code)
    return found


def build_entitlement():
    frames = []
    for service, layer in COMPLETED_LAYERS:
        frames.append(pd.DataFrame(arcgis_layer(service, layer)))
    cases = pd.concat(frames, ignore_index=True)
    cases["case"] = cases["CaseNumber"].fillna(cases["CASE_NUM"]).astype(str).str.strip()
    cases["cid"] = pd.to_numeric(cases["CaseID"], errors="coerce")
    cases["completed"] = parse_case_date(cases["CompletionDate"])
    cases["applied"] = parse_case_date(cases["AppDate"])
    cases = cases.dropna(subset=["cid", "completed", "applied"])
    cases = cases.sort_values("completed").drop_duplicates("cid", keep="first")
    cases["days"] = (cases["completed"] - cases["applied"]).dt.days
    cases = cases[cases["days"] >= 0].copy()
    cases["prefix"] = cases["case"].str.split("-").str[0].str.upper()
    cases["tokens"] = [case_tokens(c, r) for c, r in zip(cases["case"], cases["RequestType"])]

    rows = []

    def add(code, name, subset):
        if len(subset) >= MIN_CASES:
            days = subset["days"]
            rows.append(
                {
                    "code": code,
                    "plain_name": name,
                    "n_cases": int(len(subset)),
                    "p25_days": pct(days, 25),
                    "median_days": pct(days, 50),
                    "p75_days": pct(days, 75),
                    "p90_days": pct(days, 90),
                }
            )

    for code, name in PREFIX_TYPES:
        wanted = {"VTT", "TT"} if code == "VTT/TT" else {code}
        add(code, name, cases[cases["prefix"].isin(wanted)])
    for code, name, keys in SUFFIX_TYPES:
        add(code, name, cases[cases["tokens"].map(lambda t: bool(t & keys))])
    table = pd.DataFrame(rows, columns=OUTPUTS["entitlement_lookup.csv"])
    for column in ["p25_days", "median_days", "p75_days", "p90_days"]:
        table[column] = table[column].round(0).astype(int)
    return table


def load_ladbs():
    frames = []
    for dataset in LADBS_DATASETS:
        frames.append(socrata_csv("data.lacity.org", dataset, LADBS_WHERE, f"ladbs_{dataset}.csv"))
    permits = pd.concat(frames, ignore_index=True).drop_duplicates("permit_nbr")
    for column in ["submitted_date", "issue_date", "cofo_date"]:
        permits[column] = pd.to_datetime(permits[column].str[:10], errors="coerce")
    permits["middle"] = permits["permit_nbr"].str.split("-").str[1]
    permits["base"] = permits["middle"].str.endswith("0000", na=False)
    return permits


def asset_class(use_desc, work_desc):
    use = use_desc if isinstance(use_desc, str) else ""
    work = work_desc if isinstance(work_desc, str) else ""
    if re.search("apartment", use, re.I) and MIXED_WORK_RE.search(work):
        return "mixed-use"
    if MULTI_RE.search(use):
        return "multifamily"
    if HOTEL_RE.search(use):
        return "hotel"
    if OFFICE_RE.search(use):
        return "office"
    if INDUSTRIAL_RE.search(use):
        return "industrial"
    if RETAIL_RE.search(use):
        return "retail"
    return "other commercial"


def build_construction(permits):
    frame = permits[permits["base"]].copy()
    frame = frame[(frame["issue_date"] >= CONSTRUCTION_START) & (frame["issue_date"] <= CONSTRUCTION_END)]
    junk = frame["use_desc"].fillna("").map(lambda u: bool(JUNK_RE.search(u)))
    frame = frame[~junk].copy()
    frame["asset_class"] = [asset_class(u, w) for u, w in zip(frame["use_desc"], frame["work_desc"])]
    frame["has_cofo"] = frame["cofo_date"].notna()
    frame["gap_days"] = (frame["cofo_date"] - frame["issue_date"]).dt.days
    rows = []
    for name in ASSET_ORDER:
        group = frame[frame["asset_class"] == name]
        usable = group[group["gap_days"] >= MIN_COFO_GAP_DAYS]
        if len(usable) == 0:
            continue
        months = usable["gap_days"] / DAYS_PER_MONTH
        rows.append(
            {
                "asset_class": name,
                "n_permits": int(len(usable)),
                "p25_months": round(pct(months, 25), 1),
                "median_months": round(pct(months, 50), 1),
                "p75_months": round(pct(months, 75), 1),
                "share_with_cofo": round(float(group["has_cofo"].mean()), 3),
            }
        )
    return pd.DataFrame(rows, columns=OUTPUTS["construction_lookup.csv"])


def benchmark_rows(city, applied, issued, data_date):
    frame = pd.DataFrame({"applied": applied.to_numpy(), "issued": issued.to_numpy()})
    frame = frame.dropna(subset=["applied"])
    frame["days"] = (frame["issued"] - frame["applied"]).dt.days
    frame = frame[~(frame["days"] < 0)].copy()
    frame["year"] = frame["applied"].dt.year
    rows = []
    for year in BENCH_YEARS:
        group = frame[frame["year"] == year]
        if len(group) == 0:
            continue
        year_end = pd.Timestamp(year=year, month=12, day=31)
        row = {"city": city, "filing_year": year, "n_applications": int(len(group))}
        for horizon, label in [(365, "share_12m"), (730, "share_24m")]:
            if year_end + pd.Timedelta(days=horizon) <= data_date:
                row[label] = round(float(((group["days"] <= horizon) & group["issued"].notna()).mean()), 3)
            else:
                row[label] = np.nan
        issued_days = group["days"].dropna()
        row["median_days"] = round(float(issued_days.median()), 1) if len(issued_days) else np.nan
        rows.append(row)
    return rows


def build_benchmark(permits):
    data_date = max(permits["submitted_date"].max(), permits["issue_date"].max())
    work = permits["work_desc"].fillna("")
    keep = permits[permits["base"] & ~work.str.contains(PLAN_CHECK_ONLY_RE)]
    rows = benchmark_rows("Los Angeles", keep["submitted_date"], keep["issue_date"], data_date)

    seattle = socrata_csv("data.seattle.gov", SEATTLE_DATASET, SEATTLE_WHERE, "seattle_76t5-zqzr.csv")
    seattle = seattle.drop_duplicates("permitnum")
    seattle = seattle[
        (seattle["permittypedesc"] == "New")
        & (seattle["permittypemapped"] == "Building")
        & seattle["permitclass"].isin(["Commercial", "Industrial", "Institutional", "Multifamily"])
    ]
    applied = pd.to_datetime(seattle["applieddate"].str[:10], errors="coerce")
    issued = pd.to_datetime(seattle["issueddate"].str[:10], errors="coerce")
    seattle_date = max(applied.max(), issued.max())
    rows += benchmark_rows("Seattle", applied, issued, seattle_date)
    return pd.DataFrame(rows)[OUTPUTS["city_benchmark.csv"]]


def self_check():
    for name, columns in OUTPUTS.items():
        path = DATA / name
        assert path.exists(), f"missing {name}"
        table = pd.read_csv(path)
        assert list(table.columns) == columns, f"columns of {name}: {list(table.columns)}"
        assert len(table) > 0, f"empty {name}"
        assert not table.isna().any().any(), f"empty cells in {name}"
        print(f"\n{name}")
        print(table.to_string(index=False))
    benchmark = pd.read_csv(DATA / "city_benchmark.csv")
    assert set(benchmark["city"]) == {"Los Angeles", "Seattle"}
    assert benchmark.groupby("city")["filing_year"].nunique().min() == len(BENCH_YEARS)
    entitlement = pd.read_csv(DATA / "entitlement_lookup.csv")
    assert (entitlement["n_cases"] >= MIN_CASES).all()
    assert {"CPC", "DIR", "ZA"} <= set(entitlement["code"])
    print("\nself-check passed")


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    build_entitlement().to_csv(DATA / "entitlement_lookup.csv", index=False)
    permits = load_ladbs()
    build_construction(permits).to_csv(DATA / "construction_lookup.csv", index=False)
    build_benchmark(permits).to_csv(DATA / "city_benchmark.csv", index=False)
    self_check()


if __name__ == "__main__":
    sys.exit(main())
