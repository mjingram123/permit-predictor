scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=0.01,
    early_stopping=True,
    max_iter=2000,
    random_state=67,
)
nn_model.fit(X_train_scaled, y_train)
nn_days = np.expm1(nn_model.predict(X_test_scaled)).clip(0, MONTHS * 30)

day_scores["Neural Network"] = score_days(y_test, nn_days)
print("Neural Network RMSE:", round(day_scores["Neural Network"]["RMSE (days)"]))
print("Neural Network R^2 Score:", round(day_scores["Neural Network"]["R squared"], 3))
