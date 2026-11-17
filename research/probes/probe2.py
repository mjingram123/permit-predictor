import sys, re
import numpy as np, pandas as pd
d = sys.argv[1]
df = pd.concat([pd.read_csv(f"{d}/{i}.csv", dtype=str) for i in ["n3xg-rixm", "gwh9-jnip"]], ignore_index=True)
df["sub"] = pd.to_datetime(df.submitted_date); df["iss"] = pd.to_datetime(df.issue_date)
df["days"] = (df.iss - df["sub"]).dt.days; df["year"] = df["sub"].dt.year
wd = df.work_desc.fillna("").str.upper(); z = df.zone.fillna("")
mid = df.permit_nbr.str.split("-").str[1]
df["supp"] = ~mid.str.endswith("0000")
flags = {
 "case number in text": wd.str.contains(r"\b(?:CPC|DIR|ZA|APC[A-Z]*|VTT|TT|ENV|ADM|PAR)-\d{4}-\d+", regex=True),
 "TOC in text": wd.str.contains(r"\bTOC\b|TRANSIT ORIENTED", regex=True),
 "density bonus in text": wd.str.contains(r"DENSITY BONUS|12\.22[ .-]?A[ .-]?25|\bDB\b", regex=True),
 "affordable in text": wd.str.contains(r"AFFORDABLE|\bVLI\b|\bELI\b|100%", regex=True),
 "ED1 in text": wd.str.contains(r"\bED ?1\b|EXECUTIVE DIRECTIVE", regex=True),
 "zoning only / PDPP / HSAP": wd.str.contains(r"ZONING ONLY|PDPP|HSAP|PARALLEL DESIGN|PRELIMINARY ZONING|SB ?330", regex=True),
 "T or Q in zone": z.str.contains(r"\(T\)|\(Q\)|\[Q\]|\[T\]", regex=True),
 "CPIO/overlay in zone": z.str.contains(r"CPIO|-O\b|CDO|HPOZ|SN|RIO|CUGU|POD", regex=True),
 "bldg N of M": wd.str.contains(r"BLDG ?#|BUILDING ?#|\d+ OF \d+|BLDG [A-Z0-9]\b", regex=True),
 "subterranean/basement": wd.str.contains(r"SUBTERRANEAN|BASEMENT", regex=True),
 "supplemental number": df.supp,
}
print("n rows", len(df))
base = df[~df.supp]
for k, f in flags.items():
    a = base[f.loc[base.index]]; b = base[~f.loc[base.index]]
    print(f"{k:28s} share {f.mean():.2f} | issued share with {a.iss.notna().mean():.2f} without {b.iss.notna().mean():.2f} | median days with {a.days.median():.0f} without {b.days.median():.0f} (n issued with {a.days.notna().sum()})")
print()
print("case-number mentions by year:", df[flags["case number in text"]].groupby("year").size().to_dict())
print("examples:")
for t in df[flags["case number in text"]].work_desc.head(4): print("  ", t[:230])
print("zoning-only examples:")
for t in df[flags["zoning only / PDPP / HSAP"]].work_desc.head(4): print("  ", t[:230])
g = base.groupby("apn").agg(n=("permit_nbr","size"), span=("sub", lambda s: (s.max()-s.min()).days))
print("\napn groups: share of apns with >1 row", (g.n>1).mean().round(2), "; rows in multi groups", g[g.n>1].n.sum(), "; median span days of multi groups", g[g.n>1].span.median())
same = base.groupby(["apn","sub"]).size()
print("rows filed same day same apn as another row:", same[same>1].sum())
