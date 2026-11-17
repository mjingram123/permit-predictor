projects["city_floor_area"] = projects.square_footage
projects["city_height"] = projects.height
projects["city_use_filled"] = projects.use_desc.notna().astype(int)
projects["city_construction_filled"] = projects.construction.notna().astype(int)
CHEATING_INPUTS = ["city_floor_area", "city_height", "city_use_filled", "city_construction_filled"]

cheat_test, cheat_curves, cheat_model = backtest(2018, (2019, 2021), extra=CHEATING_INPUTS)
cheat_results = score_odds(cheat_test, {"Cheating model": cheat_curves["Odds model"]})

honest_scores = odds_results.xs("Odds model", level="method")["ranking score"]
cheating_scores = cheat_results.xs("Cheating model", level="method")["ranking score"]
leak_comparison = pd.DataFrame({"odds model, honest inputs": honest_scores, "cheating model": cheating_scores})
leak_comparison["fake gain"] = leak_comparison.iloc[:, 1] - leak_comparison.iloc[:, 0]
leak_comparison
