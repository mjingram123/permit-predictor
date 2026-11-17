def first_number(text_column, pattern, largest):
    found = text_column.str.extract(pattern)[0].str.replace(",", "", regex=False)
    number = pd.to_numeric(found, errors="coerce")
    return number.where((number > 0) & (number <= largest))


projects["stories"] = first_number(projects.text, r"(\d+)[- ]?STOR", 80)
projects["units"] = first_number(projects.text, r"(\d+)[- ]?(?:UNITS?\b|DU\b|DWELLING|APARTMENTS?\b)", 3000)
projects["floor_area"] = first_number(projects.text, r"([\d,]{4,9}) ?(?:SQ\.? ?FT|SF\b|S\.F\.|SQUARE)", 5000000)
projects["valuation"] = projects.valuation.where(projects.valuation > 0)
projects["filing_month"] = projects.filed.dt.month

DESCRIPTION_FLAGS = {
    "affordable_housing": r"AFFORDABLE|\bVLI\b|\bELI\b|LOW INCOME",
    "transit_incentive": r"\bTOC\b|TRANSIT ORIENTED",
    "density_bonus": r"DENSITY BONUS|12\.22",
    "underground_parking": r"SUBTERRANEAN|BASEMENT|UNDERGROUND",
    "concrete_podium": r"PODIUM|TYPE I[- ]?A\b|TYPE 1[- ]?A\b",
    "wood_frame": r"TYPE V|TYPE III|TYPE 5|TYPE 3",
    "shell_only": r"SHELL",
    "electronic_plans": r"EPLAN|E-PLAN",
}
for name, pattern in DESCRIPTION_FLAGS.items():
    projects[name] = projects.text.str.contains(pattern, regex=True).astype(int)

zone_text = projects.zone.fillna("")
projects["zone_conditions"] = zone_text.str.contains(r"\(T\)|\(Q\)|\[Q\]|\[T\]", regex=True).astype(int)
projects["zone_overlay"] = zone_text.str.contains(r"CPIO|CDO|HPOZ|-SN|RIO|CUGU|POD|-O\b", regex=True).astype(int)
zone_base = zone_text.str.extract(r"^(?:\(T\)|\(Q\)|\[Q\]|\[T\])*([A-Z]+\d*(?:\.\d)?)")[0].fillna("UNKNOWN")
common_zones = zone_base.value_counts().index[:20]
projects["zone_base"] = zone_base.where(zone_base.isin(common_zones), "OTHER")
projects["council_district"] = projects.cd.fillna("UNKNOWN").astype(str)
projects["planning_area"] = projects.apc.fillna("UNKNOWN").astype(str)

projects[["filed", "building_type", "valuation", "stories", "units", "floor_area", "buildings_filed", "zone_base", "days_to_permit"]].head()
