import sys, re, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
S = sys.argv[1]
d = pd.concat([pd.read_csv(f"{S}/verify/{i}.csv", dtype=str) for i in ["n3xg-rixm", "gwh9-jnip"]], ignore_index=True)
d["iss"] = pd.to_datetime(d.issue_date, errors="coerce")
for c in ["valuation","square_footage","height","du_changed","lat","lon"]: d[c] = pd.to_numeric(d[c], errors="coerce")
wd = d.work_desc.fillna("").str.upper(); ud = d.use_desc.fillna("").str.upper()
junk = ud.str.contains(r"CARPORT|PATIO|CANOPY|TEMPORARY|GARAGE - PRIVATE|ACCESSORY|SINGLE FAMILY|DUPLEX|FOUNDATION|FENCE|WALL|SIGN|POOL|TENT|TANK|ANTENNA|TOWER|MISCELLANEOUS|SHED|TRASH|RETAINING|GAZEBO|TRELLIS|EQUIPMENT|RECREATION ROOM", regex=True)
d = d[d.iss.notna() & ~junk & d.permit_nbr.str.split("-").str[1].str.endswith("0000") & (d.valuation > 50000) & (d.square_footage > 200)].copy()
wd = d.work_desc.fillna("").str.upper()
d["stories"] = pd.to_numeric(wd.str.extract(r"(\d+)[- ]?STOR")[0], errors="coerce")
d["use"] = d.use_desc.fillna("?").where(d.use_desc.map(d.use_desc.value_counts()) >= 30, "OTHER").astype("category")
d["construction"] = d.construction.astype("category"); d["iy"] = d.iss.dt.year
for k, p in {"subt": r"SUBTERRANEAN|BASEMENT", "podium": r"PODIUM", "afford": r"AFFORDABLE", "shell": r"SHELL", "tilt": r"TILT", "parking": r"PARKING"}.items(): d["kw_" + k] = wd.str.contains(p, regex=True).astype(int)
d["y"] = np.log(d.valuation)
num = ["square_footage","height","du_changed","lat","lon","stories","iy"] + [c for c in d.columns if c.startswith("kw_")]; cat = ["use","construction"]
tr = d[d.iy <= 2019]; te = d[d.iy >= 2020]; tr = tr[~tr.apn.isin(set(te.apn.dropna()))]
print(f"CONSTRUCTION COST (permit valuation) from building description: train {len(tr)} (issued 2010-2019), test {len(te)} (issued 2020+); median test valuation ${te.valuation.median():,.0f}")
cm = tr.groupby("use", observed=True).y.median(); pb = te.use.map(cm).astype(float).fillna(tr.y.median()).values
ppsf = (tr.valuation / tr.square_footage).groupby(tr.use, observed=True).median(); pb2 = np.log(te.use.map(ppsf).astype(float).fillna((tr.valuation / tr.square_footage).median()).values * te.square_footage.values)
X = lambda z: pd.get_dummies(z[num + cat], columns=cat, dtype=float)
Xtr = X(tr); Xte = X(te).reindex(columns=Xtr.columns, fill_value=0)
for c in ["square_footage","height","du_changed","stories"]:
    Xtr[c] = np.log1p(Xtr[c].clip(lower=0)); Xte[c] = np.log1p(Xte[c].clip(lower=0))
f = Xtr.median(); Xtr = Xtr.fillna(f).fillna(0); Xte = Xte.fillna(f).fillna(0)
lr = LinearRegression().fit(Xtr, tr.y); nn = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True, max_iter=1500, random_state=67)).fit(Xtr, tr.y)
g = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=1).fit(tr[num + cat], tr.y)
for nm, p in [("median cost for that use", pb), ("typical $/sq ft for that use x size", pb2), ("linear regression", lr.predict(Xte)), ("neural network", nn.predict(Xte)), ("gradient boosting", g.predict(te[num + cat]))]:
    ape = np.abs(np.exp(p) - te.valuation) / te.valuation
    print(f"   {nm:38s} R2 (log $) {r2_score(te.y, p):5.2f} | median % error {np.median(ape) * 100:4.0f}% | within 25% of actual: {np.mean(ape <= 0.25) * 100:3.0f}%")
