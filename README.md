# LA Permit Timeline Predictor

Odds and date ranges for a building permit on new commercial and multifamily buildings in Los Angeles, from public city data. A student project for the Marshall AI Association (MAIA).

Open `LA_Permit_Timeline_Predictor.ipynb` in Google Colab and choose Runtime, then Run all. It needs no logins and takes a few minutes.

## What is here

| Path | What it is |
|---|---|
| `LA_Permit_Timeline_Predictor.ipynb` | The notebook, generated from `notebook_src/` |
| `notebook_src/` | One file per notebook cell, in order |
| `build_notebook.py` | Assembles the notebook from `notebook_src/` |
| `data/` | Saved snapshot the notebook falls back to, plus helper tables |
| `build/` | Scripts that rebuild the helper tables in `data/` from public sources |
| `figures/` | Charts saved from the latest run |
| `research/` | Test scripts from the planning stage and the cell runner |

## Data sources

- City of Los Angeles open data, "Building Permits Submitted" (datasets `n3xg-rixm` and `gwh9-jnip`), new buildings in the Commercial and Apartment categories.
- Permit history for each building site from the same portal (`data/parcel_history.csv`).
- Los Angeles City Planning completed case layers, 2020 to 2025 (`data/entitlement_lookup.csv`).
- Seattle open data, building permits `76t5-zqzr` (`data/city_benchmark.csv`).

## Rebuilding

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install pandas numpy scikit-learn lightgbm matplotlib seaborn ipywidgets nbformat nbconvert ipykernel
```

```bash
.venv/bin/python build/parcel_history.py
```

```bash
.venv/bin/python build/lookups.py
```

```bash
.venv/bin/python build_notebook.py
```

`research/run_cells.py` runs every cell against the local `data/` folder without a notebook and saves the charts to `figures/`.
