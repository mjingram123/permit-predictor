position = 40
example = finished_test.iloc[[position]]
example_row = np.flatnonzero(test.project_id.values == example.project_id.values[0])[0]

print("--- Project ---")
display(example[["filed", "asset_class", "valuation", "stories", "units", "primary_address"]])
print(f"Description: {example.work_desc.values[0][:200]}")
print()
print(f"Actual days to permit: {y_test[position]:.0f}")
print(f"Typical wait for the building type: {baseline_days[position]:.0f}")
print(f"Linear Regression: {linear_days[position]:.0f}")
print(f"Neural Network: {nn_days[position]:.0f}")
print(f"Gradient Boosting: {boosting_days[position]:.0f}")
print()
for months in (12, 24, 36):
    happened = "yes" if y_test[position] <= months * 30 else "no"
    print(f"Odds of a permit within {months} months (Blend): {curves['Blend'][example_row, months - 1]:.0%}   Did it happen? {happened}")
