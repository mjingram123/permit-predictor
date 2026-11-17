checked_columns = ["use_desc", "square_footage", "height", "construction", "valuation", "zone", "apn", "lat", "work_desc"]
got_permit = buildings.permit_date.notna()
fill_rates = pd.DataFrame({
    "filled in, permit issued": buildings[got_permit][checked_columns].notna().mean(),
    "filled in, never issued": buildings[~got_permit][checked_columns].notna().mean(),
})
fill_rates["gap"] = fill_rates.iloc[:, 0] - fill_rates.iloc[:, 1]
fill_rates["safe to use"] = np.where(fill_rates.gap.abs() < 0.05, "yes", "no")
CITY_FILLED = list(fill_rates.index[fill_rates["safe to use"] == "no"])
fill_rates.round(2)
