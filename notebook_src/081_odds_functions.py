odds_settings = dict(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=80,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5,
    random_state=67,
    verbose=-1,
)


def waiting_months(table, waited, has_permit):
    last_month = np.minimum(waited // 30, MONTHS - 1).astype(int)
    counts = last_month + 1
    project = np.repeat(np.arange(len(table)), counts)
    months_waited = np.concatenate([np.arange(count) for count in counts])
    is_last_month = months_waited == np.repeat(last_month, counts)
    arrived = has_permit & (waited < MONTHS * 30)
    permit_came = (is_last_month & np.repeat(arrived, counts)).astype(int)
    part_of_month_seen = np.where(has_permit | (waited >= MONTHS * 30), 1.0, (waited % 30) / 30)
    exposure = np.where(is_last_month, np.repeat(part_of_month_seen, counts), 1.0)
    calendar_date = table.filed.values[project] + (months_waited * 30).astype("timedelta64[D]")
    return project, months_waited, permit_came, calendar_date, exposure


def fit_odds_model(table, waited, has_permit, extra=()):
    project, months_waited, permit_came, calendar_date, exposure = waiting_months(table, waited, has_permit)
    rows = tree_table(table, extra).iloc[project].reset_index(drop=True)
    rows["months_waited"] = months_waited
    model = lgb.LGBMClassifier(**odds_settings)
    model.fit(rows, permit_came, sample_weight=np.maximum(exposure, 0.000001))
    return model


def permit_curves(model, table, extra=()):
    features = tree_table(table, extra)
    rows = features.iloc[np.repeat(np.arange(len(features)), MONTHS)].reset_index(drop=True)
    rows["months_waited"] = np.tile(np.arange(MONTHS), len(features))
    monthly_chance = model.predict_proba(rows)[:, 1].reshape(len(features), MONTHS)
    return 1 - np.cumprod(1 - monthly_chance, axis=1)


def track_record(table, waited, has_permit, cutoff):
    project, months_waited, permit_came, calendar_date, exposure = waiting_months(table, waited, has_permit)
    last_two_years = calendar_date >= np.datetime64(cutoff - pd.Timedelta(days=730))
    building_type = table.building_type.values[project]

    def monthly_chance(chosen):
        waiting = np.bincount(months_waited[chosen], weights=exposure[chosen], minlength=MONTHS)
        permits = np.bincount(months_waited[chosen], weights=permit_came[chosen], minlength=MONTHS)
        return np.minimum(permits / np.maximum(waiting, 1), 1), waiting

    all_types, waiting = monthly_chance(last_two_years)
    all_types = pd.Series(np.where(waiting >= 5, all_types, np.nan)).ffill().fillna(0).values
    curves = {}
    for name in CATEGORY_TYPES["building_type"].categories:
        own, waiting = monthly_chance(last_two_years & (building_type == name))
        trust = np.minimum(waiting / 150, 1)
        curves[name] = 1 - np.cumprod(1 - (trust * own + (1 - trust) * all_types))
    return curves


def track_record_curves(table, waited, has_permit, cutoff, classes):
    curves = track_record(table, waited, has_permit, cutoff)
    return np.vstack([curves[name] for name in classes])


def score_odds(table, curves):
    rows = []
    for months in (12, 24, 36):
        watched_long_enough = ((DATA_DATE - table.filed).dt.days >= months * 30).values
        actual = (table.days_to_permit <= months * 30).values[watched_long_enough]
        for name, curve in curves.items():
            chance = curve[watched_long_enough, months - 1]
            rows.append({
                "permit within": f"{months} months",
                "method": name,
                "projects": int(watched_long_enough.sum()),
                "actual share": actual.mean(),
                "average odds given": chance.mean(),
                "ranking score": roc_auc_score(actual, chance),
                "odds error": np.mean((chance - actual) ** 2),
            })
    return pd.DataFrame(rows).set_index(["permit within", "method"]).round(3)


def month_reached(curves, share):
    share_of_final = curves / np.maximum(curves[:, -1:], 1e-9)
    return np.argmax(share_of_final >= share, axis=1) + 1


def typical_days(curves):
    return month_reached(curves, 0.5) * 30 - 15


def range_holds(table, curves, low, high):
    watched_four_years = ((DATA_DATE - table.filed).dt.days >= MONTHS * 30).values
    finished = (table.days_to_permit <= MONTHS * 30).values & watched_four_years
    actual = table.days_to_permit.values[finished]
    earliest = (month_reached(curves, low)[finished] - 1) * 30
    latest = month_reached(curves, high)[finished] * 30
    return np.mean((actual >= earliest) & (actual <= latest))


def backtest(last_training_year, test_years, extra=()):
    train, test, cutoff = split_by_time(last_training_year, test_years)
    waited, has_permit = as_known_on(train, cutoff)
    model = fit_odds_model(train, waited, has_permit, extra)
    curves = {
        "Track record": track_record_curves(train, waited, has_permit, cutoff, test.building_type),
        "Odds model": permit_curves(model, test, extra),
    }
    curves["Blend"] = (curves["Track record"] + curves["Odds model"]) / 2
    return test, curves, model
