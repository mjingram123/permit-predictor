import sys
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

d = sys.argv[1]
REFRESH = pd.Timestamp("2026-11-14")
df = pd.concat([pd.read_csv(f"{d}/{i}.csv", dtype=str) for i in ["n3xg-rixm", "gwh9-jnip"]], ignore_index=True)
df["sub"] = pd.to_datetime(df.submitted_date, errors="coerce")
df["iss"] = pd.to_datetime(df.issue_date, errors="coerce")
for c in ["valuation", "square_footage", "height", "du_changed", "lat", "lon"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df[df["sub"].notna()].copy()
wd = df.work_desc.fillna("").str.upper()
ud = df.use_desc.fillna("").str.upper()
df["wd"] = wd
df["supp"] = ~df.permit_nbr.str.split("-").str[1].str.endswith("0000")
df["pc_only"] = wd.str.contains(r"ZONING ONLY|PDPP|HSAP|PC ONLY|PARALLEL DESIGN|ZONING PLAN CHECK ONLY|ZONING REVIEW ONLY|PRELIMINARY ZONING|SB ?330", regex=True)
junk_use = r"CARPORT|PATIO|CANOPY|TEMPORARY|GARAGE - PRIVATE|ACCESSORY|SINGLE FAMILY|DUPLEX|FOUNDATION|FENCE|WALL|SIGN|POOL|TENT|TANK|ANTENNA|TOWER|MISCELLANEOUS|SHED|TRASH|RETAINING|GAZEBO|TRELLIS|EQUIPMENT|RECREATION ROOM"
df["junk"] = ud.str.contains(junk_use, regex=True)


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


MODE = sys.argv[3] if len(sys.argv) > 3 else "safe"
df["klass"] = [klass("", w, s) if True else None for u, w, s in zip(ud, wd, df.permit_sub_type)]
df.loc[(df.klass == "other commercial") & wd.str.contains(r"OFFICE|MEDICAL|CLINIC|BANK", regex=True), "klass"] = "office"
df.loc[(df.klass == "other commercial") & wd.str.contains(r"WAREHOUSE|MANUFACTUR|INDUSTRIAL|STORAGE|DISTRIBUTION|TILT", regex=True), "klass"] = "industrial"
df.loc[(df.klass == "other commercial") & wd.str.contains(r"RETAIL|RESTAURANT|MARKET|STORE|SHOP|CAR WASH|GAS STATION|DRIVE.?THR|DEALERSHIP|CAFE", regex=True), "klass"] = "retail"
junk_txt = r"CARPORT|PATIO|CANOPY|TRASH ENCLOSURE|\bSIGN\b|FENCE|RETAINING WALL|POOL|\bTENT\b|\bTANK\b|ANTENNA|MONOPOLE|SHADE STRUCTURE|GAZEBO|TRELLIS|STORAGE SHED|GUARD (?:SHACK|BOOTH|HOUSE)|KIOSK|MODULAR|PAY STATION|TEMPORARY|TRAILER|PERGOLA|CABANA|BLEACHER|GENERATOR|EQUIPMENT (?:ENCLOSURE|SHELTER|PAD)|RESTROOM BUILDING|PREFAB|BUNGALOW|PORTABLE|RELOCATABLE|CABIN|PUMP HOUSE|DUGOUT|PLAY ?GROUND|LUNCH SHELTER"
df["junk"] = wd.str.contains(junk_txt, regex=True) & ~wd.str.contains(r"\d+[- ]?STOR(?:Y|IES).{0,60}(?:UNIT|APARTMENT|OFFICE|RETAIL|HOTEL|WAREHOUSE)", regex=True)
df = df[~df.pc_only & ~df.supp].copy()
wd = df.wd
df["key"] = df.apn.fillna("ADDR:" + df.primary_address.fillna("?").str.upper().str.replace(r"\s+\d.*$", "", regex=True))
df = df.sort_values(["key", "sub"]).reset_index(drop=True)
GAP = int(sys.argv[2]) if len(sys.argv) > 2 else 365
newproj = (df.key != df.key.shift(1)) | ((df["sub"] - df["sub"].shift(1)).dt.days > GAP)
df["pid"] = newproj.cumsum()
core = df[~df.junk]
g = core.groupby("pid")
main = core.loc[core.groupby("pid").valuation.idxmax().dropna().astype(int)] if False else None
P = g.agg(key=("key", "first"), start=("sub", "min"), first_issue=("iss", "min"), n_rows=("permit_nbr", "size"),
          val_max=("valuation", "max"), val_sum=("valuation", "sum"), sqft_max=("square_footage", "max"), sqft_sum=("square_footage", "sum"),
          height=("height", "max"), du=("du_changed", "sum"), lat=("lat", "median"), lon=("lon", "median"),
          has_pc_only=("pc_only", "max"), n_supp=("supp", "sum"), zone=("zone", "first"), cd=("cd", "first"), apc=("apc", "first"),
          text=("wd", lambda s: " | ".join(s)), n_days_filed=("sub", "nunique"))
order = core.sort_values("valuation", ascending=False).drop_duplicates("pid").set_index("pid")
P["klass"] = order.klass.reindex(P.index)
P["construction"] = order.construction.reindex(P.index)
P["sub_type"] = order.permit_sub_type.reindex(P.index)
P["T"] = (P.first_issue - P.start).dt.days
P.loc[P["T"] < 0, "T"] = 0
P["year"] = P.start.dt.year
print(f"gap rule {GAP} days: rows {len(df)} -> projects {len(P)} (dropped junk-only projects: {df.pid.nunique() - len(P)})")
print("rows per project:", P.n_rows.describe().round(2).to_dict())
print("ever issued share: rows %.2f | projects %.2f" % (df[~df.junk & ~df.supp].iss.notna().mean(), P.first_issue.notna().mean()))
old = P[P.year <= 2018]
print("projects filed 2010-2018: ever issued %.2f ; rows filed 2010-2018 (non-junk, non-revision): %.2f ; plan-check-only rows excluded too: %.2f" % (
    old.first_issue.notna().mean(), df[(df["sub"].dt.year <= 2018) & ~df.junk & ~df.supp].iss.notna().mean(),
    df[(df["sub"].dt.year <= 2018) & ~df.junk & ~df.supp & ~df.pc_only].iss.notna().mean()))
print("by class (projects 2010+): n, issued share, median days")
print(P.groupby("klass").agg(n=("T", "size"), issued=("first_issue", lambda s: round(s.notna().mean(), 2)), med=("T", "median")).to_string())

t = P.text
P["stories"] = pd.to_numeric(t.str.extract(r"(\d+)[- ]?STOR")[0], errors="coerce")
P["units_txt"] = pd.to_numeric(t.str.extract(r"(\d+)[- ]?(?:UNIT|DU\b|APARTMENT)")[0], errors="coerce")
P["zone_pref"] = P.zone.fillna("").str.extract(r"^(?:\(T\)|\(Q\)|\[Q\]|\[T\])*([A-Z]+\d*(?:\.\d)?)")[0]
P["zone_tq"] = P.zone.fillna("").str.contains(r"\(T\)|\(Q\)|\[Q\]|\[T\]", regex=True).astype(int)
P["zone_overlay"] = P.zone.fillna("").str.contains(r"CPIO|CDO|HPOZ|SN|RIO|CUGU|POD|-O", regex=True).astype(int)
kws = {"affordable": r"AFFORDABLE|\bVLI\b|\bELI\b|100%", "toc": r"\bTOC\b|TRANSIT ORIENTED", "db": r"DENSITY BONUS|12\.22", "subterr": r"SUBTERRANEAN|BASEMENT",
       "podium": r"PODIUM|TYPE I[- ]?A", "case": r"\b(?:CPC|DIR|ZA|APC[A-Z]*|VTT|TT|ENV)-\d{4}", "shell": r"SHELL", "demo": r"DEMO", "parking": r"PARKING",
       "t3": r"TYPE III", "t5": r"TYPE V", "eplan": r"EPLAN|E-PLAN", "phase": r"PHASE", "ed1": r"\bED ?1\b", "modular": r"MODULAR|PREFAB", "tilt": r"TILT",
       "retail": r"RETAIL|RESTAURANT", "hotel": r"HOTEL", "school": r"SCHOOL|CLASSROOM", "mezz": r"MEZZANINE", "sprink": r"SPRINKLER", "sb35": r"SB ?35|AB ?2011|SB ?423"}
for k, pat in kws.items():
    P["kw_" + k] = t.str.contains(pat, regex=True).astype(int)
P["month"] = P.start.dt.month
P["textlen"] = t.str.len()
for c in ["val_max", "val_sum", "sqft_max", "sqft_sum", "du", "units_txt"]:
    P[c] = P[c].where(P[c] > 0)
cat = ["klass", "zone_pref", "cd", "apc", "sub_type"]
for c in cat:
    P[c] = P[c].astype("category")
P["sqft_txt"] = pd.to_numeric(t.str.extract(r"([\d,]{4,9}) ?(?:SQ\.? ?FT|SF\b|S\.F\.|SQUARE)")[0].str.replace(",", ""), errors="coerce")
base_num = ["val_max", "val_sum", "lat", "lon", "month", "n_rows", "n_days_filed", "stories", "units_txt", "sqft_txt", "zone_tq", "zone_overlay"] + (["textlen"] if MODE != "notext" else []) + (["kw_" + k for k in kws] if MODE != "notext" else [])
if MODE == "notext":
    base_num = [c for c in base_num if c not in ("stories", "units_txt", "sqft_txt")]
P["has_pc_only"] = P.has_pc_only.astype(int)
HOR = [360, 720, 1080]
NI = 48


def km_curve(T, E, grid):
    T = np.asarray(T, float)
    E = np.asarray(E, bool)
    out = []
    s = 1.0
    ts = np.sort(np.unique(T[E]))
    surv = {}
    for u in ts:
        at = (T >= u).sum()
        dd = ((T == u) & E).sum()
        s *= 1 - dd / at
        surv[u] = s
    keys = np.array(sorted(surv))
    vals = np.array([surv[k] for k in keys])
    for gday in grid:
        i = np.searchsorted(keys, gday, side="right") - 1
        out.append(1 - (vals[i] if i >= 0 else 1.0))
    return np.array(out)


def frozen(tr, cutoff):
    T = tr["T"].values.astype(float)
    E = tr.first_issue.notna().values & (tr.first_issue.values <= np.datetime64(cutoff))
    obs = np.where(E, T, (cutoff - tr.start).dt.days.values)
    return obs, E


def expand(X, obs, E, feats):
    k_last = np.minimum((obs // 30).astype(int), NI - 1)
    reps = k_last + 1
    idx = np.repeat(np.arange(len(X)), reps)
    k = np.concatenate([np.arange(r) for r in reps])
    out = X[feats].iloc[idx].reset_index(drop=True)
    out["k"] = k
    ev = E & (obs < NI * 30)
    y = ((k == np.repeat(k_last, reps)) & np.repeat(ev, reps)).astype(int)
    return out, y


def cum_from_hazard(model, X, feats):
    n = len(X)
    idx = np.repeat(np.arange(n), NI)
    k = np.tile(np.arange(NI), n)
    Z = X[feats].iloc[idx].reset_index(drop=True)
    Z["k"] = k
    h = model.predict_proba(Z)[:, 1].reshape(n, NI)
    S = np.cumprod(1 - h, axis=1)
    return 1 - S


def evaluate(cut_year, test_years, use_recent_feat=True, label=""):
    cutoff = pd.Timestamp(f"{cut_year + 1}-01-01")
    tr = P[P.start < cutoff].copy()
    te = P[(P.year >= test_years[0]) & (P.year <= test_years[1])].copy()
    tr = tr[~tr.key.isin(set(te.key))]
    obs, E = frozen(tr, cutoff)
    print(f"\n===== {label} train filed <= {cut_year} (labels frozen at {cutoff.date()}): {len(tr)} projects, {E.sum()} issued by cutoff | test filed {test_years}: {len(te)} projects")
    grid_days = np.arange(1, NI + 1) * 30
    rec = (tr.start >= cutoff - pd.Timedelta(days=3 * 365)).values
    base = {}
    for kname in P.klass.cat.categories:
        m = (tr.klass == kname).values & rec
        base[kname] = km_curve(obs[m], E[m], grid_days) if m.sum() >= 20 and E[m].sum() >= 5 else None
    allk = km_curve(obs[rec], E[rec], grid_days)
    allk_full = km_curve(obs, E, grid_days)
    Pb = np.vstack([base[k] if base[k] is not None else allk for k in te.klass.astype(str)])
    feats = base_num + cat
    tr2 = tr.copy()
    te2 = te.copy()
    if use_recent_feat:
        r12 = {k: (base[k][11] if base[k] is not None else allk[11]) for k in base}
        r24 = {k: (base[k][23] if base[k] is not None else allk[23]) for k in base}
    Xtr, ytr = expand(tr2, obs, E, feats)
    params = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1)
    Ph = np.zeros((len(te2), NI))
    for seed in [1, 2, 3]:
        mdl = lgb.LGBMClassifier(random_state=seed, **params).fit(Xtr, ytr)
        Ph += cum_from_hazard(mdl, te2, feats) / 3
    Tt = te["T"].values
    rows = []
    for hday in HOR:
        j = hday // 30 - 1
        y = (te.first_issue.notna().values & (Tt <= hday)).astype(int)
        okk = (REFRESH - te.start).dt.days.values >= hday
        y, pb, ph = y[okk], Pb[okk, j], Ph[okk, j]
        ratio = pb.mean() / max(ph.mean(), 1e-9)
        b0 = np.mean((pb - y) ** 2)
        b1 = np.mean((ph - y) ** 2)
        rows.append((hday, okk.sum(), y.mean(), pb.mean(), ph.mean(), b0, b1, 1 - b1 / b0, roc_auc_score(y, pb) if len(set(pb)) > 1 else np.nan, roc_auc_score(y, ph)))
    print("horizon  n   actual  base_pred  model_pred  Brier_base  Brier_model  skill  AUC_base  AUC_model")
    for r in rows:
        print("%5d  %5d  %.3f   %.3f      %.3f       %.4f      %.4f      %+.3f   %.3f     %.3f" % r)
    print("MID-REVIEW: for projects still waiting after L days, chance of a permit in the next 12 months")
    for L in [180, 360, 540, 720]:
        a, b2 = L // 30 - 1, (L + 360) // 30 - 1
        still = ~(te.first_issue.notna().values & (Tt <= L)) & ((REFRESH - te.start).dt.days.values >= L + 360)
        yy = (te.first_issue.notna().values & (Tt > L) & (Tt <= L + 360))[still].astype(int)
        pm = ((Ph[:, b2] - Ph[:, a]) / np.clip(1 - Ph[:, a], 1e-6, None))[still]
        pbb = ((Pb[:, b2] - Pb[:, a]) / np.clip(1 - Pb[:, a], 1e-6, None))[still]
        print("   waited %4d days: n %4d | actual %.2f | track record says %.2f | model says %.2f | AUC track record %.2f, model %.2f | Brier track %.3f, model %.3f" % (
            L, still.sum(), yy.mean(), pbb.mean(), pm.mean(), roc_auc_score(yy, pbb) if len(set(np.round(pbb, 6))) > 1 else float("nan"), roc_auc_score(yy, pm), np.mean((pbb - yy) ** 2), np.mean((pm - yy) ** 2)))
    uncond = (te.first_issue.notna().values & (Tt <= 360)).mean()
    print("   for comparison, chance of a permit in the first 12 months from filing: %.2f" % uncond)
    j = 23
    y = (te.first_issue.notna().values & (Tt <= 720)).astype(int)
    ph = Ph[:, j]
    q = pd.qcut(ph, 5, duplicates="drop")
    cal = pd.DataFrame({"p": ph, "y": y}).groupby(q, observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).round(2)
    print("calibration at 24 months (5 bins):")
    print(cal.to_string())
    bc = pd.DataFrame({"k": te.klass.astype(str).values, "y": y, "pb": Pb[:, j], "ph": ph})
    print("per class at 24 months: n, actual, base pred, model pred, Brier base, Brier model, AUC model")
    for kname, gdf in bc.groupby("k"):
        auc = roc_auc_score(gdf.y, gdf.ph) if gdf.y.nunique() > 1 else float("nan")
        print(f"  {kname:18s} {len(gdf):5d}  {gdf.y.mean():.2f}  {gdf.pb.mean():.2f}  {gdf.ph.mean():.2f}  {np.mean((gdf.pb - gdf.y) ** 2):.3f}  {np.mean((gdf.ph - gdf.y) ** 2):.3f}  {auc:.2f}")
    fin = te.first_issue.notna().values & (Tt <= NI * 30)
    med_model = np.array([grid_days[np.searchsorted(row / max(row[-1], 1e-9), 0.5)] if row[-1] > 0 else np.nan for row in Ph])
    med_base = np.array([grid_days[np.searchsorted(row / max(row[-1], 1e-9), 0.5)] for row in Pb])
    print("finished test projects only (n=%d): MAE model %.0f days, MAE class baseline %.0f days; rank corr model %.2f, baseline %.2f" % (
        fin.sum(), np.nanmean(np.abs(med_model[fin] - Tt[fin])), np.mean(np.abs(med_base[fin] - Tt[fin])),
        pd.Series(med_model[fin]).corr(pd.Series(Tt[fin]), method="spearman"), pd.Series(med_base[fin]).corr(pd.Series(Tt[fin]), method="spearman")))
    lo = np.array([grid_days[min(np.searchsorted(row / max(row[-1], 1e-9), 0.1), NI - 1)] for row in Ph])
    hi = np.array([grid_days[min(np.searchsorted(row / max(row[-1], 1e-9), 0.9), NI - 1)] for row in Ph])
    q1 = np.array([grid_days[min(np.searchsorted(row / max(row[-1], 1e-9), 0.25), NI - 1)] for row in Ph])
    q3 = np.array([grid_days[min(np.searchsorted(row / max(row[-1], 1e-9), 0.75), NI - 1)] for row in Ph])
    print("range coverage on finished test projects: 10-90 range holds %.2f (target 0.80), 25-75 range holds %.2f (target 0.50); median 10-90 width %.0f days" % (
        np.mean((Tt[fin] >= lo[fin] - 30) & (Tt[fin] <= hi[fin])), np.mean((Tt[fin] >= q1[fin] - 30) & (Tt[fin] <= q3[fin])), np.median(hi - lo)))
    imp = pd.Series(mdl.feature_importances_, index=feats + ["k"]).sort_values(ascending=False).head(14)
    print("top features:", ", ".join(f"{i}({v})" for i, v in imp.items()))


evaluate(2018, (2019, 2021), label="A")
evaluate(2015, (2016, 2018), label="B")
evaluate(2021, (2022, 2023), label="C")
