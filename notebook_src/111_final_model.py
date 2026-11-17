all_waited, all_have_permit = as_known_on(projects, DATA_DATE + pd.Timedelta(days=1))
final_model = fit_odds_model(projects, all_waited, all_have_permit)
todays_track_record = track_record(projects, all_waited, all_have_permit, DATA_DATE)

newest_first = projects.sort_values("filed", ascending=False)
todays_pace = {name: newest_first[name].dropna().iloc[0] for name in ["recent_quick_share", "recent_typical_wait"]}
typical_site_permits = projects.n_prior_permits_5y.median()
uses_electronic_plans = int(newest_first.electronic_plans.head(300).mean() > 0.5)


def most_common(values):
    return values.value_counts().index[0]


projects["zip5"] = projects.zip_code.fillna("").astype(str).str[:5]
zip_table = projects[projects.zip5.str.len() == 5].groupby("zip5").agg(
    lat=("lat", "median"),
    lon=("lon", "median"),
    council_district=("council_district", most_common),
    planning_area=("planning_area", most_common),
    zone_base=("zone_base", most_common),
)
citywide = {"lat": projects.lat.median(), "lon": projects.lon.median(), "council_district": most_common(projects.council_district), "planning_area": most_common(projects.planning_area), "zone_base": most_common(projects.zone_base)}


def describe_project(asset_class, valuation, stories, units, floor_area, buildings_filed, zip_code, zone, ticked):
    place = zip_table.loc[zip_code].to_dict() if zip_code in zip_table.index else dict(citywide)
    row = {name: 0.0 for name in NUMBERS}
    row.update({
        "valuation": valuation if valuation > 0 else np.nan,
        "stories": stories if stories > 0 else np.nan,
        "units": units if units > 0 else np.nan,
        "floor_area": floor_area if floor_area > 0 else np.nan,
        "buildings_filed": max(buildings_filed, 1),
        "lat": place["lat"],
        "lon": place["lon"],
        "filing_month": DATA_DATE.month,
        "n_prior_permits_5y": typical_site_permits,
        "electronic_plans": uses_electronic_plans,
        "recent_quick_share": todays_pace["recent_quick_share"],
        "recent_typical_wait": todays_pace["recent_typical_wait"],
        "asset_class": asset_class,
        "zone_base": place["zone_base"] if zone == "Typical for the ZIP code" else zone,
        "council_district": place["council_district"],
        "planning_area": place["planning_area"],
    })
    for name in ticked:
        row[name] = 1.0
    return pd.DataFrame([row])


def blended_curve(row):
    model_curve = permit_curves(final_model, row)[0]
    record_curve = todays_track_record[row.asset_class.iloc[0]]
    return (model_curve + record_curve) / 2, model_curve, record_curve


def similar_projects(row, how_many=25):
    pool = projects[(projects.asset_class == row.asset_class.iloc[0]) & (projects.filed >= DATA_DATE - pd.Timedelta(days=8 * 365)) & projects.valuation.notna()]
    target_value = row.valuation.iloc[0] if pd.notna(row.valuation.iloc[0]) else pool.valuation.median()
    distance = (np.log(pool.valuation) - np.log(target_value)).abs()
    for name in ["units", "stories"]:
        if pd.notna(row[name].iloc[0]):
            gap = (np.log1p(pool[name]) - np.log1p(row[name].iloc[0])).abs()
            distance = distance + 0.5 * gap.fillna(gap.median())
    return pool.loc[distance.sort_values().index[:how_many]]


print(f"Final model trained on {len(projects):,} projects through {DATA_DATE.date()}")
