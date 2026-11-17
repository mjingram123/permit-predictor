periods = {"2013 to 2015": (2012, (2013, 2015)), "2016 to 2018": (2015, (2016, 2018)), "2019 to 2021": (2018, (2019, 2021)), "2022 to 2023": (2021, (2022, 2023))}
backtests = {"2019 to 2021": (test, curves)}
report_rows = []
for name, (last_training_year, test_years) in periods.items():
    if name not in backtests:
        period_test, period_curves, period_model = backtest(last_training_year, test_years)
        backtests[name] = (period_test, period_curves)
    period_test, period_curves = backtests[name]
    scores = score_odds(period_test, period_curves).loc["24 months"]
    report_rows.append({
        "projects filed": name,
        "projects": len(period_test),
        "permit within 24 months": scores.loc["Blend", "actual share"],
        "odds error, track record": scores.loc["Track record", "odds error"],
        "odds error, blend": scores.loc["Blend", "odds error"],
        "ranking, track record": scores.loc["Track record", "ranking score"],
        "ranking, blend": scores.loc["Blend", "ranking score"],
        "80% range held": range_holds(period_test, period_curves["Blend"], 0.1, 0.9),
        "50% range held": range_holds(period_test, period_curves["Blend"], 0.25, 0.75),
    })
report_card = pd.DataFrame(report_rows).set_index("projects filed").round(3)
report_card
