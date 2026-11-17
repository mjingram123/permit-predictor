MONTHS = 48
LOG_SCALED = ["valuation", "stories", "units", "floor_area", "buildings_filed", "n_prior_permits_5y", "recent_typical_wait"]


def split_by_time(last_training_year, test_years):
    cutoff = pd.Timestamp(year=last_training_year + 1, month=1, day=1)
    test = projects[projects.filing_year.between(test_years[0], test_years[1])]
    train = projects[(projects.filed < cutoff) & ~projects.site.isin(set(test.site))]
    return train, test, cutoff


def as_known_on(table, cutoff):
    has_permit = (table.permit_date.notna() & (table.permit_date < cutoff)).values
    waited = np.where(has_permit, table.days_to_permit, (cutoff - table.filed).dt.days).astype(float)
    return np.clip(waited, 0, None), has_permit


def tree_table(table, extra=()):
    out = table[NUMBERS + list(extra)].astype(float).copy()
    for name in CATEGORIES:
        out[name] = table[name].astype(CATEGORY_TYPES[name])
    return out


def plain_table(table, fill_values):
    numbers = table[NUMBERS].astype(float).copy()
    for name in LOG_SCALED:
        numbers[name] = np.log1p(numbers[name])
    numbers = numbers.fillna(fill_values)
    labels = pd.get_dummies(pd.DataFrame({name: table[name].astype(CATEGORY_TYPES[name]) for name in ["asset_class", "planning_area"]}), dtype=float)
    return pd.concat([numbers, labels], axis=1)


def fill_values_from(table):
    numbers = table[NUMBERS].astype(float).copy()
    for name in LOG_SCALED:
        numbers[name] = np.log1p(numbers[name])
    return numbers.median().fillna(0)


def score_days(actual, predicted):
    return {
        "RMSE (days)": np.sqrt(mean_squared_error(actual, predicted)),
        "typical miss (days)": np.median(np.abs(actual - predicted)),
        "R squared": r2_score(actual, predicted),
    }


train, test, cutoff = split_by_time(2018, (2019, 2021))
train_waited, train_has_permit = as_known_on(train, cutoff)
finished_train = train[train_has_permit]
finished_test = test[test.days_to_permit <= MONTHS * 30]

print(f"Training projects: {len(train):,} ({len(finished_train):,} had a permit by {cutoff.date()})")
print(f"Test projects: {len(test):,} ({len(finished_test):,} have a permit so far)")
