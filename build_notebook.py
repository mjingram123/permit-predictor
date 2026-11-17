import pathlib
import sys

import nbformat

root = pathlib.Path(__file__).resolve().parent
include_team = "--with-team" in sys.argv
target = root / ("deliverable" if include_team else "") / "LA_Permit_Timeline_Predictor.ipynb"
target.parent.mkdir(exist_ok=True)

cells = []
for path in sorted((root / "notebook_src").iterdir()):
    if path.name == "001_team.md" and not include_team:
        continue
    text = path.read_text().strip()
    if path.suffix == ".md":
        cells.append(nbformat.v4.new_markdown_cell(text))
    elif path.suffix == ".py":
        cells.append(nbformat.v4.new_code_cell(text))

notebook = nbformat.v4.new_notebook(cells=cells)
notebook.metadata = {
    "colab": {"provenance": [], "toc_visible": True},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
}
nbformat.write(notebook, target)
print(f"wrote {target} with {len(cells)} cells")
