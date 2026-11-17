older = load_permits("n3xg-rixm", "2010 to 2019", "permits_2010_2019.csv.gz", 8000)
newer = load_permits("gwh9-jnip", "2020 to today", "permits_2020_present.csv.gz", 5000)
permits = pd.concat([older, newer], ignore_index=True)

site_history = load_helper("parcel_history.csv")
entitlement_times = load_helper("entitlement_lookup.csv")
construction_times = load_helper("construction_lookup.csv").rename(columns={"asset_class": "building_type"})
city_benchmark = load_helper("city_benchmark.csv")

permits[["permit_nbr", "primary_address", "permit_sub_type", "submitted_date", "issue_date", "valuation", "work_desc"]].head()
