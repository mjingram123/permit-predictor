import sys, re, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
S = sys.argv[1]
def load(paths):
    d = pd.concat([pd.read_csv(p, dtype=str) for p in paths], ignore_index=True)
    d["sub"] = pd.to_datetime(d.submitted_date, errors="coerce"); d["iss"] = pd.to_datetime(d.issue_date, errors="coerce")
    for c in ["valuation","square_footage","height","du_changed","lat","lon"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    d["wd"] = d.work_desc.fillna("").str.upper()
    d["supp"] = ~d.permit_nbr.str.split("-").str[1].str.endswith("0000")
    d["zone_pref"] = d.zone.fillna("").str.extract(r"^(?:\(T\)|\(Q\)|\[Q\]|\[T\])*([A-Z]+\d*(?:\.\d)?)")[0]
    d["year"] = d["sub"].dt.year; d["month"] = d["sub"].dt.month
    d["days"] = (d.iss - d["sub"]).dt.days
    return d[d["sub"].notna()]
P = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=1)

print("################ B: ALL plan-checked permits on commercial and apartment buildings (filing-day fields only)")
cols = ["permit_nbr","apn","zone","permit_type","permit_sub_type","submitted_date","issue_date","valuation","lat","lon","work_desc","cd","apc","du_changed","square_footage","height"]
nb = [pd.read_csv(f"{S}/verify/{i}.csv", dtype=str, usecols=lambda c: c in cols + ["business_unit"]) for i in ["n3xg-rixm","gwh9-jnip"]]
nb = [x[x.business_unit == "Regular Plan Check"] for x in nb]
for i, x in enumerate(nb): x.to_csv(f"{S}/verify/_nb{i}.csv", index=False)
d = load([f"{S}/probes/broader/n3xg-rixm_broad.csv", f"{S}/probes/broader/gwh9-jnip_broad.csv", f"{S}/verify/_nb0.csv", f"{S}/verify/_nb1.csv"])
d = d[~d.supp & ~d.wd.str.contains(r"ZONING ONLY|PDPP|HSAP|PC ONLY|SB ?330", regex=True)].copy()
print("rows", len(d)); print(d.permit_type.value_counts().head(8).to_dict())
kws = {"ti": r"TENANT IMPROVEMENT|\bT\.?I\.?\b", "cou": r"CHANGE OF USE", "add": r"ADDITION", "seis": r"SEISMIC|SOFT.?STORY|RETROFIT", "sign": r"\bSIGN", "solar": r"SOLAR|PHOTOVOLTAIC", "demo": r"DEMO",
       "rest": r"RESTAURANT|KITCHEN", "office": r"OFFICE", "retail": r"RETAIL|STORE", "apt": r"APARTMENT|UNIT", "struct": r"STRUCTURAL", "nonstruct": r"NON.?STRUCTURAL", "interior": r"INTERIOR", "roof": r"ROOF",
       "ada": r"\bADA\b|ACCESSIB|DISABLED", "antenna": r"ANTENNA|WIRELESS|CELL", "storage": r"STORAGE RACK|RACK", "afford": r"AFFORDABLE", "toc": r"\bTOC\b", "subt": r"SUBTERRANEAN|BASEMENT", "new": r"\bNEW\b", "repair": r"REPAIR|REPLACE", "fire": r"FIRE DAMAGE|FIRE", "eplan": r"EPLAN", "medical": r"MEDICAL|CLINIC|HOSPITAL", "school": r"SCHOOL", "hotel": r"HOTEL", "convert": r"CONVERT|CONVERSION", "adu": r"\bADU\b|ACCESSORY DWELLING"}
for k, p in kws.items(): d["kw_" + k] = d.wd.str.contains(p, regex=True).astype(int)
d["textlen"] = d.wd.str.len()
d["stories"] = pd.to_numeric(d.wd.str.extract(r"(\d+)[- ]?STOR")[0], errors="coerce")
d["units_txt"] = pd.to_numeric(d.wd.str.extract(r"(\d+)[- ]?UNIT")[0], errors="coerce")
d["sqft_txt"] = pd.to_numeric(d.wd.str.extract(r"([\d,]{3,9}) ?(?:SQ\.? ?FT|SF\b|S\.F\.)")[0].str.replace(",", ""), errors="coerce")
cat = ["permit_type","permit_sub_type","zone_pref","cd","apc"]
for c in cat: d[c] = d[c].astype("category")
num = ["valuation","lat","lon","month","textlen","stories","units_txt","sqft_txt"] + ["kw_" + k for k in kws]
te = d[(d.year >= 2019) & (d.year <= 2021)]
for hname, hday, cut in [("3 months", 90, 2018), ("6 months", 180, 2018), ("12 months", 365, 2017)]:
    tr = d[(d.year <= cut) & ~d.apn.isin(set(te.apn.dropna()))]
    ytr = (tr.days <= hday).fillna(False).astype(int); yte = (te.days <= hday).fillna(False).astype(int)
    rec = tr[tr.year >= cut - 2]
    rate = ((rec.days <= hday).fillna(False)).groupby(rec.permit_type, observed=True).mean()
    pb = te.permit_type.map(rate).astype(float).fillna(ytr.mean()).values
    m = lgb.LGBMClassifier(**P).fit(tr[num + cat], ytr); ph = m.predict_proba(te[num + cat])[:, 1]
    b0, b1 = np.mean((pb - yte) ** 2), np.mean((ph - yte) ** 2)
    print(f"permit within {hname}: train {len(tr)} test {len(te)} | actual rate {yte.mean():.2f} | baseline (rate by permit type, last 3y) AUC {roc_auc_score(yte, pb):.3f} | model AUC {roc_auc_score(yte, ph):.3f} | Brier skill vs baseline {1 - b1 / b0:+.3f}")
    isnew = (te.permit_type == "Bldg-New").values
    print(f"    new buildings only inside this model: n {isnew.sum()} AUC {roc_auc_score(yte[isnew], ph[isnew]):.3f} | alterations only: AUC {roc_auc_score(yte[(te.permit_type=='Bldg-Alter/Repair').values], ph[(te.permit_type=='Bldg-Alter/Repair').values]):.3f}")
tr = d[(d.year <= 2017) & d.days.notna() & (d.days >= 0) & (d.iss < "2019-01-01") & ~d.apn.isin(set(te.apn.dropna()))]; t2 = te[te.days.notna() & (te.days >= 0)]
med = tr[tr.year >= 2015].groupby("permit_type", observed=True).days.median(); pb = t2.permit_type.map(med).astype(float).fillna(tr.days.median()).values
m = lgb.LGBMRegressor(**P).fit(tr[num + cat], np.log1p(tr.days)); pl = m.predict(t2[num + cat]); ph = np.expm1(pl)
print(f"days to permit, issued only: train {len(tr)} test {len(t2)} | baseline MAE {np.mean(np.abs(pb - t2.days)):.0f} d, model MAE {np.mean(np.abs(ph - t2.days)):.0f} d | R2 on log days: baseline {r2_score(np.log1p(t2.days), np.log1p(pb)):.2f}, model {r2_score(np.log1p(t2.days), pl):.2f} | rank corr model {pd.Series(ph).corr(pd.Series(t2.days.values), method='spearman'):.2f} | actual median {t2.days.median():.0f} d")

print("\n################ E: CONSTRUCTION TIME, permit issued -> certificate of occupancy (new commercial and apartment buildings)")
e = load([f"{S}/verify/n3xg-rixm.csv", f"{S}/verify/gwh9-jnip.csv", f"{S}/verify/b6ii-mhed.csv"])
e["cofo"] = pd.to_datetime(e.cofo_date, errors="coerce")
ud = e.use_desc.fillna("").str.upper()
junk = ud.str.contains(r"CARPORT|PATIO|CANOPY|TEMPORARY|GARAGE - PRIVATE|ACCESSORY|SINGLE FAMILY|DUPLEX|FOUNDATION|FENCE|WALL|SIGN|POOL|TENT|TANK|ANTENNA|TOWER|MISCELLANEOUS|SHED|TRASH|RETAINING|GAZEBO|TRELLIS|EQUIPMENT|RECREATION ROOM", regex=True)
e = e[~e.supp & ~junk & e.iss.notna()].copy()
e["iy"] = e.iss.dt.year
print("share of issued new buildings with a certificate of occupancy, by permit year:", e[e.iy.between(2008, 2024)].groupby("iy").cofo.apply(lambda s: round(s.notna().mean(), 2)).to_dict())
e["cdays"] = (e.cofo - e.iss).dt.days
e = e[e.cdays.notna() & (e.cdays >= 30)]
def kl(u, w):
    if re.search(r"HOTEL|MOTEL", u): return "hotel"
    if re.search(r"APARTMENT|DWELLING - MULTIPLE|CONDO", u): return "mixed-use" if re.search(r"MIXED.?USE|RETAIL|COMMERCIAL", w) else "multifamily"
    if re.search(r"OFFICE|BANK|CLINIC|MEDICAL", u): return "office"
    if re.search(r"WAREHOUSE|MANUFACTUR|STORAGE|INDUSTRIAL", u): return "industrial"
    if re.search(r"RETAIL|GROCERY|RESTAURANT|CAFE|CAR WASH|GAS|AUTO|STORE|MARKET", u): return "retail"
    return "other commercial"
e["klass"] = [kl(u, w) for u, w in zip(e.use_desc.fillna("").str.upper(), e.wd)]
e["stories"] = pd.to_numeric(e.wd.str.extract(r"(\d+)[- ]?STOR")[0], errors="coerce")
for k, p in {"subt": r"SUBTERRANEAN|BASEMENT", "podium": r"PODIUM|TYPE I[- ]?A", "t3": r"TYPE III", "t5": r"TYPE V", "afford": r"AFFORDABLE", "shell": r"SHELL", "tilt": r"TILT", "parking": r"PARKING"}.items(): e["kw_" + k] = e.wd.str.contains(p, regex=True).astype(int)
e["imonth"] = e.iss.dt.month
ecat = ["klass","zone_pref","cd","construction","permit_sub_type"]
for c in ecat: e[c] = e[c].astype("category")
enum = ["valuation","square_footage","height","du_changed","lat","lon","stories","imonth"] + [c for c in e.columns if c.startswith("kw_")]
print(e.groupby("klass", observed=True).cdays.describe(percentiles=[.25,.5,.75]).round(0)[["count","25%","50%","75%"]].to_string())
for (a, b, c2, d2) in [(2010, 2016, 2017, 2019), (2000, 2013, 2014, 2016), (2010, 2018, 2019, 2021)]:
    tr = e[e.iy.between(a, b) & (e.cofo < f"{c2}-01-01")]; te2 = e[e.iy.between(c2, d2)]; tr = tr[~tr.apn.isin(set(te2.apn.dropna()))]
    ytr, yte = np.log(tr.cdays), np.log(te2.cdays)
    cm = tr.groupby("klass", observed=True).cdays.median(); pb = te2.klass.map(cm).astype(float).fillna(tr.cdays.median()).values
    g = lgb.LGBMRegressor(**{**P, "num_leaves": 15}).fit(tr[enum + ecat], ytr); pg = g.predict(te2[enum + ecat])
    Xtr = pd.get_dummies(tr[enum + ["klass"]], columns=["klass"], dtype=float); Xte = pd.get_dummies(te2[enum + ["klass"]], columns=["klass"], dtype=float).reindex(columns=Xtr.columns, fill_value=0)
    for c in ["valuation","square_footage","height","du_changed","stories"]:
        Xtr[c] = np.log1p(Xtr[c].clip(lower=0)); Xte[c] = np.log1p(Xte[c].clip(lower=0))
    fill = Xtr.median(); Xtr = Xtr.fillna(fill).fillna(0); Xte = Xte.fillna(fill).fillna(0)
    lr = LinearRegression().fit(Xtr, ytr); pl = lr.predict(Xte)
    nn = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, early_stopping=True, max_iter=1000, random_state=67)).fit(Xtr, ytr); pn = nn.predict(Xte)
    print(f"permits issued {a}-{b} -> test {c2}-{d2}: train {len(tr)} test {len(te2)} | actual median {te2.cdays.median():.0f} d")
    for nm, p in [("class median baseline", np.log(pb)), ("linear regression", pl), ("neural network", pn), ("gradient boosting", pg)]:
        print(f"    {nm:24s} R2(log) {r2_score(yte, p):5.2f} | MAE {np.mean(np.abs(np.exp(p) - te2.cdays)):4.0f} d | median abs err {np.median(np.abs(np.exp(p) - te2.cdays)):4.0f} d | rank corr {pd.Series(p).corr(pd.Series(yte.values), method='spearman'):.2f}")
    imp = pd.Series(g.feature_importances_, index=enum + ecat).sort_values(ascending=False).head(8); print("    top features:", ", ".join(imp.index))
