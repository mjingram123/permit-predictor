def city_pace(projects):
    ordered = projects.sort_values("filed")
    filed_days = ordered.filed.values.astype("datetime64[D]").astype(int)
    quick = (ordered.days_to_permit <= 360).fillna(False).values.astype(float)
    running_quick = np.concatenate([[0.0], np.cumsum(quick)])
    window_start = np.searchsorted(filed_days, filed_days - 1080, side="left")
    window_end = np.searchsorted(filed_days, filed_days - 360, side="right")
    count = window_end - window_start
    recent_share = np.where(count >= 30, (running_quick[window_end] - running_quick[window_start]) / np.maximum(count, 1), np.nan)

    issued = projects[projects.permit_date.notna()].sort_values("permit_date")
    issue_days = issued.permit_date.values.astype("datetime64[D]").astype(int)
    waits = issued.days_to_permit.values
    year_start = np.searchsorted(issue_days, filed_days - 365, side="left")
    year_end = np.searchsorted(issue_days, filed_days, side="left")
    recent_wait = np.array([np.median(waits[a:b]) if b - a >= 30 else np.nan for a, b in zip(year_start, year_end)])
    return pd.DataFrame({"recent_quick_share": recent_share, "recent_typical_wait": recent_wait}, index=ordered.index)


projects[["recent_quick_share", "recent_typical_wait"]] = city_pace(projects)

NUMBERS = [
    "valuation", "stories", "units", "floor_area", "buildings_filed", "lat", "lon", "filing_month",
    "zone_conditions", "zone_overlay", "prior_demo_3y", "prior_grading_3y", "prior_newbldg_5y",
    "prior_newbldg_issued", "n_prior_permits_5y", "recent_quick_share", "recent_typical_wait",
] + list(DESCRIPTION_FLAGS)
CATEGORIES = ["building_type", "zone_base", "council_district", "planning_area"]
CATEGORY_TYPES = {name: pd.CategoricalDtype(sorted(projects[name].unique())) for name in CATEGORIES}

print(f"{len(NUMBERS) + len(CATEGORIES)} inputs per project, all known on filing day")
