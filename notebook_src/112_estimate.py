TICK_BOXES = {
    "affordable_housing": "Includes affordable housing",
    "transit_incentive": "Uses the transit incentive program (TOC)",
    "density_bonus": "Uses a density bonus",
    "underground_parking": "Underground parking",
    "concrete_podium": "Concrete podium or Type I construction",
    "wood_frame": "Wood frame (Type III or V)",
    "shell_only": "Shell and core only",
    "zone_conditions": "Zoning carries Q or T conditions",
    "zone_overlay": "Site is in an overlay district",
    "prior_demo_3y": "Demolition permit on the site in the last 3 years",
    "prior_grading_3y": "Grading permit on the site in the last 3 years",
    "prior_newbldg_5y": "Earlier new-building application on the site",
}


def month_label(months):
    return f"month {months} (about {(DATA_DATE + pd.DateOffset(months=int(months))).strftime('%b %Y')})"


def print_estimate(asset_class, valuation, stories, units, floor_area, buildings_filed, zip_code, zone, ticked, planning_approval):
    row = describe_project(asset_class, valuation, stories, units, floor_area, buildings_filed, zip_code, zone, ticked)
    blend, model_curve, record_curve = blended_curve(row)
    if zip_code not in zip_table.index:
        print(f"ZIP code {zip_code} has no past projects in our data, so a typical Los Angeles location is used.")
        print()

    print(f"NEW {asset_class.upper()} PROJECT FILED TODAY")
    print()
    print("Chance of having the building permit")
    for months in (12, 24, 36):
        print(f"   within {months} months: {blend[months - 1]:.0%}")
    print(f"   within 4 years:   {blend[-1]:.0%}")
    print()
    print("If the permit comes, when")
    labels = {0.25: "optimistic (1 in 4 are faster)", 0.5: "middle", 0.75: "pessimistic (1 in 4 are slower)"}
    for share, label in labels.items():
        print(f"   {label}: {month_label(month_reached(blend[None, :], share)[0])}")
    print()
    print("Where the 24-month number comes from")
    print(f"   track record for {asset_class}, last two years: {record_curve[23]:.0%}")
    print(f"   odds model for this project: {model_curve[23]:.0%}")

    if ticked:
        print()
        print("How each ticked box moves the 24-month odds")
        for name in ticked:
            without = row.copy()
            without[name] = 0.0
            change = (blend[23] - blended_curve(without)[0][23]) * 100
            print(f"   {TICK_BOXES[name]}: {change:+.0f} points")

    neighbors = similar_projects(row)
    old_enough = neighbors[(DATA_DATE - neighbors.filed).dt.days >= 720]
    with_permit = neighbors[neighbors.days_to_permit.notna()]
    print()
    print(f"The {len(neighbors)} most similar {asset_class} projects filed in the last 8 years")
    print(f"   {int((old_enough.days_to_permit <= 720).sum())} of the {len(old_enough)} filed at least two years ago had a permit within 24 months")
    if len(with_permit):
        print(f"   those with a permit waited {with_permit.days_to_permit.quantile(0.25):.0f} to {with_permit.days_to_permit.quantile(0.75):.0f} days (middle half), typically {with_permit.days_to_permit.median():.0f}")
    closest = neighbors.head(5)[["filed", "primary_address", "valuation", "stories", "units", "days_to_permit"]].copy()
    closest["filed"] = closest.filed.dt.date
    closest["days_to_permit"] = closest.days_to_permit.map(lambda days: "still waiting" if pd.isna(days) else f"{days:.0f}")
    display(closest)

    if planning_approval != "None (by-right project)":
        approval = entitlement_times[entitlement_times.plain_name == planning_approval].iloc[0]
        print()
        print(f"Before the permit: {planning_approval}")
        print(f"   typically {approval.median_days / 30:.0f} months; the middle half of cases took {approval.p25_days / 30:.0f} to {approval.p75_days / 30:.0f} months ({approval.n_cases:.0f} completed cases, 2020 to 2025)")
        print("   plan check can overlap with it, so do not simply add the two")
    build = construction_times[construction_times.asset_class == asset_class].iloc[0]
    print()
    print("After the permit: construction")
    print(f"   typically {build.median_months:.0f} months from permit to certificate of occupancy; the middle half took {build.p25_months:.0f} to {build.p75_months:.0f} months")

    months = np.arange(1, MONTHS + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(months, blend * 100, color="#1f77b4", linewidth=2.5, label="This project (blend)")
    plt.plot(months, record_curve * 100, color="gray", linestyle="--", label=f"Track record, {asset_class}")
    plt.xlabel("Months after filing")
    plt.ylabel("Chance of having the permit (%)")
    plt.title("Chance of having the building permit by each month")
    plt.ylim(0, 100)
    plt.legend()
    plt.show()
