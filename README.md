````markdown
https://irispredictionsvm.streamlit.app/

````markdown
# Iris SVM Decision Boundary Visualizer

This repository contains a simple notebook and supporting files to train Support Vector Classifiers (SVC) on the Iris dataset and visualize decision boundaries for pairs of features (no PCA). The notebook experiments with three SVC kernels: linear, polynomial and RBF.

Contents
--------
- `task.ipynb` - Jupyter notebook that loads `Iris.csv`, trains SVCs on the full 4-feature data and on every 2-feature pair, prints metrics, and produces decision-boundary plots for each pair and kernel.
- `Iris.csv` - Iris dataset CSV used by the notebook.
- `app.py` - (placeholder) a Flask-style app file in the repo root (if present).

Quick start
-----------
1. Create and activate a Python environment (recommended Python 3.10+). On Windows (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install required packages:

```powershell
pip install -r requirements.txt
```

3. Open the notebook in VS Code or Jupyter and run cells from the top. The new visualization cells are near the end of the notebook and will:
	- Print accuracy and macro-F1 for every 2-feature pair and kernel.
	- Show decision-boundary plots for each pair (one row per pair, 3 plots for linear/poly/rbf).

Notes
-----
- The visualization cells train classifiers on raw 2-feature data (no PCA) so plots reflect real-feature axes.
- Plotting uses a 300x300 grid. If rendering is slow, reduce grid resolution in the plotting cell (e.g. 200x200).
- If plots look crowded, run only a subset of feature pairs by modifying the `pairs` list in the plotting cell.

Troubleshooting
---------------
- If the notebook raises ImportError for any package, ensure `requirements.txt` is installed in the same environment the notebook kernel uses.
- If the notebook can't find `Iris.csv`, confirm the file is at the repository root (`d:\decessiontreesvm\Iris.csv`) and the path in the notebook matches.

Suggested next steps
--------------------
- Save decision-boundary plots to PNG files instead of inline display for reporting.
- Add an option to evaluate and save trained 2-feature models.
- Build a small Streamlit app to let you interactively choose feature pairs and kernels.

If you want, I can update `requirements.txt` with pinned versions or generate a small Streamlit app next.
