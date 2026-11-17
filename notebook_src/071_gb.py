boosting_settings = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5,
    random_state=67,
    verbose=-1,
)
boosting_model = lgb.LGBMRegressor(**boosting_settings)
boosting_model.fit(tree_table(finished_train), y_train)
boosting_days = np.expm1(boosting_model.predict(tree_table(finished_test))).clip(0, MONTHS * 30)

day_scores["Gradient Boosting"] = score_days(y_test, boosting_days)
day_results = pd.DataFrame(day_scores).T
day_results.round({"RMSE (days)": 0, "typical miss (days)": 0, "R squared": 3})
