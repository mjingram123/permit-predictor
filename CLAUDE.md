# LA Permit Timeline Predictor

MAIA (Marshall AI Association) team project. Poster due 2026-11-24, demo day 2026-12-04.

- The notebook is generated. Edit the cell files in `notebook_src/` (name order; `.md` text, `.py` code), then run `.venv/bin/python build_notebook.py` (public copy at the repo root) and `.venv/bin/python build_notebook.py --with-team` (hand-in copy in `deliverable/`, with the team cell). Never edit an `.ipynb` by hand.
- `notebook_src/001_team.md` and `deliverable/` are gitignored on purpose: teammate names stay out of the public repo.
- Team rules for all code: no comments, no docstrings, no emojis, never name the project "V" anything. Plain English in every text cell.
- Test with `.venv/bin/python research/run_cells.py` (runs every cell against local `data/`, saves charts to `figures/`). Then execute the notebook with nbconvert, then Run all in Colab from `https://colab.research.google.com/github/mjingram123/permit-predictor/blob/main/LA_Permit_Timeline_Predictor.ipynb`.
- The notebook reads helper tables from this repo's `data/` folder on GitHub, so push data changes before testing in Colab.
- Never use `use_desc`, `square_footage`, `height` or `construction` as model inputs: the city fills them in during review, so they leak the outcome. They appear only in the cheating-model demo.
- Still-waiting projects count a part-observed month as a part month (`exposure` in `081_odds_functions.py`). Dropping or fully counting that month changes the scores noticeably.
- Every score is a time-split backtest with labels frozen at the cutoff. Do not add a random split.
