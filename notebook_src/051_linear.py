recent_finished = finished_train[finished_train.permit_date >= cutoff - pd.Timedelta(days=3 * 365)]
typical_by_class = recent_finished.groupby("building_type").days_to_permit.median()
baseline_days = finished_test.building_type.map(typical_by_class).fillna(recent_finished.days_to_permit.median()).values

fill_values = fill_values_from(finished_train)
X_train = plain_table(finished_train, fill_values)
X_test = plain_table(finished_test, fill_values)
y_train = np.log1p(finished_train.days_to_permit)
y_test = finished_test.days_to_permit.values

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_days = np.expm1(linear_model.predict(X_test)).clip(0, MONTHS * 30)

day_scores = {"Baseline (typical wait)": score_days(y_test, baseline_days), "Linear Regression": score_days(y_test, linear_days)}
print("Baseline RMSE (days):", round(day_scores["Baseline (typical wait)"]["RMSE (days)"]))
print("Linear Regression RMSE (days):", round(day_scores["Linear Regression"]["RMSE (days)"]))
print("Linear Regression R² Score:", round(day_scores["Linear Regression"]["R²"], 3))
