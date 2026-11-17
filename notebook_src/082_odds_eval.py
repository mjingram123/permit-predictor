test, curves, odds_model = backtest(2018, (2019, 2021))
odds_results = score_odds(test, curves)

finished = (test.days_to_permit <= MONTHS * 30).values
odds_days = typical_days(curves["Odds model"])[finished]
day_scores["Odds model (middle estimate)"] = score_days(y_test, odds_days)
day_results = pd.DataFrame(day_scores).T

print("Projects filed 2019 to 2021, trained on what was known on January 1, 2019")
display(odds_results)
