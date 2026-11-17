import sys
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

d = sys.argv[1]
df = pd.concat([pd.read_csv(f"{d}/{i}.csv", dtype=str) for i in ["b6ii-mhed", "n3xg-rixm", "gwh9-jnip"]], ignore_index=True)
print("raw rows", len(df))
df["sub"] = pd.to_datetime(df.submitted_date, errors="coerce")
df["iss"] = pd.to_datetime(df.issue_date, errors="coerce")
df["cofo"] = pd.to_datetime(df.cofo_date, errors="coerce")
df["refresh"] = pd.to_datetime(df.refresh_time, errors="coerce").max()
df["days"] = (df.iss - df["sub"]).dt.days
for c in ["valuation", "square_footage", "height", "du_changed", "lat", "lon"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["year"] = df["sub"].dt.year
mid = df.permit_nbr.str.split("-").str[1]
df["supp"] = ~mid.str.endswith("0000")
wd = df.work_desc.fillna("").str.upper()
ud = df.use_desc.fillna("").str.upper()
df["zoning_only"] = wd.str.contains(r"ZONING ONLY|PDPP|HSAP|ZONING PLAN CHECK ONLY|ZONING REVIEW ONLY|PRELIMINARY ZONING", regex=True)
junk_use = r"CARPORT|PATIO|CANOPY|TEMPORARY|GARAGE|ACCESSORY|SINGLE FAMILY|DUPLEX|FOUNDATION|FENCE|WALL|SIGN|POOL|TENT|TANK|ANTENNA|TOWER|MISCELLANEOUS|SHED|TRASH|DWELLING - SINGLE|RETAINING|ADU|ACCESSORY DWELLING|SOLAR|GAZEBO|TRELLIS|EQUIPMENT|PRIVATE GARAGE"
df["junk"] = ud.str.contains(junk_use, regex=True)
print("use_desc top 45:")
print(ud.value_counts().head(45).to_string())


def klass(u, w, s):
    if re.search(r"MIXED.?USE|LIVE.?WORK|APARTMENT.{0,40}(RETAIL|COMMERCIAL)|(RETAIL|COMMERCIAL).{0,40}APARTMENT|RESIDENTIAL OVER", w):
        return "mixed-use"
    if re.search(r"HOTEL|MOTEL", u) or re.search(r"HOTEL|GUEST ?ROOMS", w):
        return "hotel"
    if re.search(r"APARTMENT|DWELLING - MULTIPLE|CONDO|MULTI", u) or s == "Apartment":
        return "multifamily"
    if re.search(r"OFFICE|BANK|CLINIC|MEDICAL", u):
        return "office"
    if re.search(r"WAREHOUSE|MANUFACTUR|STORAGE|INDUSTRIAL|FACTORY", u):
        return "industrial"
    if re.search(r"RETAIL|GROCERY|RESTAURANT|CAFE|CAR WASH|GAS|AUTO|STORE|MARKET|SHOPPING", u):
        return "retail"
    return "other commercial"


df["klass"] = [klass(u, w, s) for u, w, s in zip(ud, wd, df.permit_sub_type)]
clean = df[~df.supp & ~df.zoning_only & ~df.junk & df["sub"].notna()].copy()
print("\nclean rows", len(clean), "issued", clean.iss.notna().sum())
clean["era"] = pd.cut(clean.year, [1900, 2009, 2019, 2030], labels=["pre2010", "2010-19", "2020+"])
t = clean.groupby(["klass", "era"], observed=True).agg(n=("days", "size"), issued=("iss", lambda s: s.notna().sum()), med=("days", "median")).unstack("era")
print(t.to_string())
print("\nclass medians (issued, 2010+):")
c10 = clean[(clean.year >= 2010) & clean.days.notna() & (clean.days >= 0)]
print(c10.groupby("klass").days.describe(percentiles=[.25, .5, .75, .9]).round(0).to_string())
print("\nby year (clean): n, issued share, median days of issued")
print(clean[clean.year >= 2000].groupby("year").agg(n=("days", "size"), issued=("iss", lambda s: round(s.notna().mean(), 2)), med=("days", "median")).to_string())
print("\nfill rates 2010+ clean:")
c = clean[clean.year >= 2010]
print(c[["valuation", "square_footage", "height", "du_changed", "apn", "lat", "zone", "use_desc", "construction", "cofo_date"]].notna().mean().round(2).to_string())
print("zero valuation share", (c.valuation.fillna(0) == 0).mean().round(3))
print("rows sharing apn with another clean row", c.apn.duplicated(keep=False).mean().round(2))
print("business_unit", c.business_unit.value_counts().to_dict())

m = clean[clean.days.notna() & (clean.days >= 0)].copy()
m["logd"] = np.log1p(m.days)
m["stories"] = pd.to_numeric(wd.loc[m.index].str.extract(r"(\d+)[- ]?STORY")[0], errors="coerce")
m["units_txt"] = pd.to_numeric(wd.loc[m.index].str.extract(r"(\d+)[- ]?UNIT")[0], errors="coerce")
m["zone_pref"] = m.zone.fillna("").str.extract(r"^\(?[TQ]?\)?\[?Q?\]?([A-Z]+\d*(?:\.\d)?)")[0]
kw = ["AFFORDABLE", "TOC", "DENSITY BONUS", "SUBTERRANEAN", "BASEMENT", "PARKING", "TYPE I", "TYPE III", "TYPE V", "EPLAN", "SHELL", "DEMO", "PODIUM", "ED1", "ED 1", "MODULAR", "TILT", "SPRINKLER", "PHASE", "BLDG", "SB 35", "100%"]
for k in kw:
    m["kw_" + k] = wd.loc[m.index].str.contains(k, regex=False).astype(int)
m["month"] = m["sub"].dt.month
m["t"] = (m["sub"] - pd.Timestamp("1990-01-01")).dt.days
m = m.sort_values("sub")
m["roll_med"] = m.days.shift(1).rolling(300, min_periods=50).median()
num = ["valuation", "square_footage", "height", "du_changed", "lat", "lon", "stories", "units_txt", "t", "month"] + ["kw_" + k for k in kw]
cat = ["klass", "zone_pref", "cd", "apc", "construction", "permit_sub_type"]
for cc in cat:
    m[cc] = m[cc].astype("category")


def run(train, test, label):
    print(f"\n=== {label}: train {len(train)} test {len(test)}")
    yt = test.days.values
    out = {}
    out["global median"] = np.full(len(test), train.days.median())
    cm = train.groupby("klass", observed=True).days.median()
    out["class median"] = test.klass.map(cm).astype(float).fillna(train.days.median()).values
    rec = train[train["sub"] >= train["sub"].max() - pd.Timedelta(days=1095)]
    rm = rec.groupby("klass", observed=True).days.median()
    out["class median last 3y"] = test.klass.map(rm).astype(float).fillna(rec.days.median()).values
    Xtr = pd.get_dummies(train[num + ["klass"]], columns=["klass"]).fillna(train[num].median())
    Xte = pd.get_dummies(test[num + ["klass"]], columns=["klass"]).reindex(columns=Xtr.columns, fill_value=0).fillna(train[num].median())
    Xtr = Xtr.fillna(0)
    Xte = Xte.fillna(0)
    for c2 in ["valuation", "square_footage", "du_changed", "units_txt"]:
        Xtr[c2] = np.log1p(Xtr[c2].clip(lower=0))
        Xte[c2] = np.log1p(Xte[c2].clip(lower=0))
    lr = LinearRegression().fit(Xtr, train.logd)
    out["linear regression (log)"] = np.expm1(lr.predict(Xte))
    sc = StandardScaler().fit(Xtr)
    nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=600, random_state=67, early_stopping=True).fit(sc.transform(Xtr), train.logd)
    out["neural net (log)"] = np.expm1(nn.predict(sc.transform(Xte)))
    g = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=67)
    g.fit(train[num + cat], train.days)
    out["lightgbm median"] = g.predict(test[num + cat])
    num2 = [x for x in num if x != "t"]
    g2 = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=67)
    g2.fit(train[num2 + cat + ["roll_med"]], train.days)
    out["lightgbm median, rolling wait not date"] = g2.predict(test[num2 + cat + ["roll_med"]])
    for k, p in out.items():
        mae = np.mean(np.abs(yt - p))
        mdae = np.median(np.abs(yt - p))
        rho = pd.Series(p).corr(pd.Series(yt), method="spearman") if np.std(p) > 0 else float("nan")
        print(f"{k:42s} MAE {mae:6.0f}  median abs err {mdae:6.0f}  rank corr {rho:5.2f}")
    lo = lgb.LGBMRegressor(objective="quantile", alpha=0.1, n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, verbose=-1, random_state=67).fit(train[num2 + cat + ["roll_med"]], train.days)
    hi = lgb.LGBMRegressor(objective="quantile", alpha=0.9, n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, verbose=-1, random_state=67).fit(train[num2 + cat + ["roll_med"]], train.days)
    a, b = lo.predict(test[num2 + cat + ["roll_med"]]), hi.predict(test[num2 + cat + ["roll_med"]])
    print("raw 10-90 range coverage", round(np.mean((yt >= a) & (yt <= b)), 2), "median width", round(np.median(b - a)))
    best = out["lightgbm median, rolling wait not date"]
    r = pd.DataFrame({"k": test.klass.values, "e": np.abs(yt - best), "b": np.abs(yt - out["class median last 3y"])})
    print(r.groupby("k", observed=True).agg(n=("e", "size"), model_mae=("e", "mean"), baseline_mae=("b", "mean")).round(0).to_string())
    imp = pd.Series(g2.feature_importances_, index=num2 + cat + ["roll_med"]).sort_values(ascending=False).head(12)
    print("top features:", ", ".join(f"{i}({v})" for i, v in imp.items()))


run(m[(m.year >= 2010) & (m.year <= 2018)], m[(m.year >= 2019) & (m.year <= 2021)], "train 2010-2018 -> test 2019-2021 (issued only)")
run(m[(m.year >= 1996) & (m.year <= 2018)], m[(m.year >= 2019) & (m.year <= 2021)], "train 1996-2018 -> test 2019-2021 (issued only)")
run(m[(m.year >= 2010) & (m.year <= 2015)], m[(m.year >= 2016) & (m.year <= 2018)], "train 2010-2015 -> test 2016-2018 (issued only)")
