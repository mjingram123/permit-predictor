def summarize(table, by):
    return table.groupby(by).agg(
        projects=("project_id", "size"),
        permit_within_12_months=("days_to_permit", lambda days: (days <= 360).mean()),
        permit_within_24_months=("days_to_permit", lambda days: (days <= 720).mean()),
        typical_days_when_issued=("days_to_permit", "median"),
    ).round(2)


two_years_old = projects[(DATA_DATE - projects.filed).dt.days >= 720]
by_class = summarize(two_years_old, "building_type")
by_year = summarize(two_years_old, "filing_year")
display(by_class)
display(by_year)
