import ast
import os
import pathlib
import sys

os.environ["MPLBACKEND"] = "Agg"
root = pathlib.Path(__file__).resolve().parent.parent
stop_after = sys.argv[1] if len(sys.argv) > 1 else "999"
history = root / "data" / "parcel_history.csv"
if not history.exists():
    import pandas as pd
    raw = pd.concat([pd.read_csv(root / "data" / name, dtype=str, usecols=["permit_nbr"]) for name in ["permits_2010_2019.csv.gz", "permits_2020_present.csv.gz"]])
    for column in ["prior_demo_3y", "prior_grading_3y", "prior_newbldg_5y", "prior_newbldg_issued", "n_prior_permits_5y"]:
        raw[column] = 0
    raw.to_csv(root / "data" / "parcel_history_stub.csv", index=False)
    print("WARNING: parcel_history.csv missing, using zeros")

namespace = {"__name__": "__main__"}
for path in sorted((root / "notebook_src").glob("*.py")):
    if path.name[:3] > stop_after:
        break
    source = path.read_text()
    source = source.replace('"https://raw.githubusercontent.com/mjingram123/permit-predictor/main/data/"', repr(str(root / "data") + "/"))
    source = source.replace("attempts=3", "attempts=0")
    if not history.exists():
        source = source.replace('"parcel_history.csv"', '"parcel_history_stub.csv"')
    source = source.replace("from IPython.display import display", "display = lambda thing: print(thing.to_string() if hasattr(thing, 'to_string') else thing)")
    print(f"\n----- {path.name}")
    tree = ast.parse(source)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, str(path), "exec"), namespace)
        value = eval(compile(last, str(path), "eval"), namespace)
        if value is not None:
            print(value.to_string() if hasattr(value, "to_string") else value)
    else:
        exec(compile(tree, str(path), "exec"), namespace)
    if "plt" in namespace:
        plt = namespace["plt"]
        for number in plt.get_fignums():
            plt.figure(number).savefig(root / "figures" / f"{path.stem}_{number}.png", dpi=200, bbox_inches="tight")
        plt.close("all")
extra = root / "research" / "after_cells.py"
if len(sys.argv) > 2 and extra.exists():
    exec(compile(extra.read_text(), str(extra), "exec"), namespace)
