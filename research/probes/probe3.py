import sys, numpy as np, pandas as pd
d = sys.argv[1]
df = pd.concat([pd.read_csv(f"{d}/{i}.csv", dtype=str) for i in ["n3xg-rixm", "gwh9-jnip"]], ignore_index=True)
df["sub"] = pd.to_datetime(df.submitted_date); df["iss"] = pd.to_datetime(df.issue_date)
df["issued"] = df.iss.notna()
wd = df.work_desc.fillna("").str.upper()
df["supp"] = ~df.permit_nbr.str.split("-").str[1].str.endswith("0000")
df["pc_only"] = wd.str.contains(r"ZONING ONLY|PDPP|HSAP|PC ONLY|PARALLEL DESIGN|ZONING PLAN CHECK ONLY|ZONING REVIEW ONLY|PRELIMINARY ZONING|SB ?330", regex=True)
b = df[~df.supp]
print("FILL RATE by issued status (non-revision rows): column, issued rows, never-issued rows")
for c in ["use_desc","use_code","square_footage","height","construction","du_changed","valuation","zone","apn","lat","cd","apc","work_desc","type_lat_lon","ev","solar","hl","ct"]:
    print(f"  {c:16s} {b[b.issued][c].notna().mean():.2f}  {b[~b.issued][c].notna().mean():.2f}")
print("\nstatus_desc of never-issued rows by filing era:")
b2 = b[~b.issued].copy(); b2["era"] = np.where(b2["sub"].dt.year<=2018,"2010-18",np.where(b2["sub"].dt.year<=2022,"2019-22","2023+"))
print(b2.groupby("era").status_desc.value_counts().groupby(level=0).head(8).to_string())
print("\nnever-issued rows filed 2010-2018: days from filing to last status change (status_date):")
x = b2[b2.era=="2010-18"]; dd = (pd.to_datetime(x.status_date)-x["sub"]).dt.days
print(dd.describe(percentiles=[.25,.5,.75,.9]).round(0).to_dict())
print("\nPC-only rows: n", b.pc_only.sum(), "issued share", b[b.pc_only].issued.mean().round(3))
b = b.sort_values(["apn","sub"])
pc = b[b.pc_only & b.apn.notna()]
others = b[~b.pc_only & b.apn.notna()]
og = others.groupby("apn")
res = []
for apn, s, in zip(pc.apn, pc["sub"]):
    if apn in og.groups:
        o = others.loc[og.groups[apn]]
        gaps = (o["sub"] - s).dt.days
        after = gaps[gaps >= -30]
        res.append((True, after.min() if len(after) else np.nan, o.issued.any()))
    else:
        res.append((False, np.nan, False))
r = pd.DataFrame(res, columns=["has_sibling","gap_to_next","sibling_issued"])
print("PC-only rows with a non-PC-only sibling on same parcel: %.2f ; sibling ever issued: %.2f" % (r.has_sibling.mean(), r.sibling_issued.mean()))
print("gap in days from PC-only filing to next sibling filing:", r.gap_to_next.describe(percentiles=[.25,.5,.75,.9]).round(0).to_dict())
print("\nexamples of blank use_desc rows (issued?, text):")
for _, row in b[b.use_desc.isna()].sample(6, random_state=1).iterrows(): print("  ", row.issued, "|", row.permit_sub_type, "|", str(row.work_desc)[:150])
