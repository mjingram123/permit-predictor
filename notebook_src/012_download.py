CITY_DATA = "https://data.lacity.org/resource/{}.csv"
SNAPSHOT = "https://raw.githubusercontent.com/mjingram123/permit-predictor/main/data/"
NEW_BUILDINGS = "permit_type='Bldg-New' AND permit_sub_type in ('Commercial','Apartment')"


def fetch_text(url, attempts=3, timeout=60):
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None


def load_permits(dataset_id, years, snapshot_name, fewest_rows):
    query = urllib.parse.urlencode({"$where": NEW_BUILDINGS, "$limit": 50000})
    text = fetch_text(CITY_DATA.format(dataset_id) + "?" + query)
    if text is not None:
        table = pd.read_csv(io.StringIO(text), dtype=str)
        if len(table) >= fewest_rows and "submitted_date" in table.columns:
            print(f"Filed {years}: {len(table):,} applications from the live city data")
            return table
    table = pd.read_csv(SNAPSHOT + snapshot_name, dtype=str)
    print(f"Filed {years}: {len(table):,} applications from the saved snapshot")
    return table


def load_helper(name):
    return pd.read_csv(SNAPSHOT + name, dtype={"permit_nbr": str})
