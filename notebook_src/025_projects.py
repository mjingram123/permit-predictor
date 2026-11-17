HISTORY_COLUMNS = ["prior_demo_3y", "prior_grading_3y", "prior_newbldg_5y", "prior_newbldg_issued", "n_prior_permits_5y"]
buildings = buildings.merge(site_history[["permit_nbr"] + HISTORY_COLUMNS], on="permit_nbr", how="left")
buildings[HISTORY_COLUMNS] = buildings[HISTORY_COLUMNS].fillna(0)


def site_of(table):
    street = table.primary_address.fillna("UNKNOWN").str.upper().str.replace(r"\s+\d[\d\s-]*$", "", regex=True)
    return table.apn.where(table.apn.notna(), "ADDRESS " + street)


buildings["site"] = site_of(buildings)
buildings = buildings.sort_values(["site", "filed"]).reset_index(drop=True)
starts_new_project = (buildings.site != buildings.site.shift()) | ((buildings.filed - buildings.filed.shift()).dt.days > 30)
buildings["project_id"] = starts_new_project.cumsum()

largest_first = buildings.sort_values("valuation", ascending=False).drop_duplicates("project_id").set_index("project_id")
grouped = buildings.groupby("project_id")
projects = grouped.agg(
    site=("site", "first"),
    filed=("filed", "min"),
    permit_date=("permit_date", "min"),
    occupancy_date=("occupancy_date", "min"),
    buildings_filed=("permit_nbr", "size"),
    valuation=("valuation", "sum"),
    lat=("lat", "median"),
    lon=("lon", "median"),
    text=("text", " | ".join),
    prior_demo_3y=("prior_demo_3y", "max"),
    prior_grading_3y=("prior_grading_3y", "max"),
    prior_newbldg_5y=("prior_newbldg_5y", "max"),
    prior_newbldg_issued=("prior_newbldg_issued", "max"),
    n_prior_permits_5y=("n_prior_permits_5y", "max"),
)
for column in ["permit_nbr", "primary_address", "zip_code", "building_type", "zone", "cd", "apc", "work_desc"] + CITY_FILLED:
    projects[column] = largest_first[column].reindex(projects.index)

early_reviews = permits[~permits.is_revision & permits.is_plan_check_only].copy()
early_reviews["site"] = site_of(early_reviews)
candidates = projects.reset_index()[["project_id", "site", "filed"]].merge(early_reviews[["site", "filed"]].rename(columns={"filed": "review_filed"}), on="site")
candidates = candidates[(candidates.review_filed < candidates.filed) & (candidates.review_filed >= candidates.filed - pd.Timedelta(days=4 * 365))]
first_review = candidates.groupby("project_id").review_filed.min().reindex(projects.index)

issued_on_filing_day = (projects.permit_date - projects.filed).dt.days <= 1
clock_unknown = issued_on_filing_day & first_review.isna()
projects["filed"] = first_review.fillna(projects.filed)
projects = projects[~clock_unknown].copy()

projects["days_to_permit"] = (projects.permit_date - projects.filed).dt.days.clip(lower=0)
projects["filing_year"] = projects.filed.dt.year
projects = projects.reset_index()

print(f"{len(buildings):,} applications grouped into {len(projects):,} projects")
print(f"{int(first_review.notna().sum())} projects had an earlier plan-check-only filing, so their clock starts there")
print(f"{int(clock_unknown.sum())} projects dropped: permit recorded on the filing day and no earlier filing found")
print(f"{projects.permit_date.notna().mean():.0%} of all projects have a permit so far")
