# Breast Cancer EDA Dashboard

A dark, premium two-column Streamlit analytics dashboard built from the provided `Breast_Cancer.csv` dataset. It uses a navy card system with cyan, teal, purple, pink, and blue Plotly accents while keeping the source column names and values available for exploration.

## What is included

- Compact Overview with dynamic KPIs, patient status, stage, survival, demographics, tumor/node analysis, correlation, and source records.
- Patient demographics, cancer stage, survival, tumor/lymph node, correlation, data explorer, and insights pages.
- Global sidebar filters for age, 6th stage, race, status, grade, estrogen status, and progesterone status.
- Interactive Plotly charts with hover details, zoom, pan, and responsive legends.
- Searchable data table and filtered CSV downloads.
- Caching for source-data loading and an empty-state experience for restrictive filters.

## Run locally

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app expects the exact source filename `Breast_Cancer.csv` beside `app.py`.

## Data handling note

The original CSV is loaded from the exact filename `Breast_Cancer.csv` and is never modified. The dashboard follows the project notebook's data-cleaning step by removing only the single exact duplicate from the in-memory analytics view, producing 4,023 unique patient records. It does not automatically remove valid medical outliers. The source data contains 4,024 rows and no missing values, so the original CSV remains available for auditing.

The EDA notebook found with the source files was an unrelated e-commerce analysis notebook. This dashboard therefore uses the breast-cancer CSV columns directly rather than treating the notebook's e-commerce outputs as breast-cancer findings.

All insights are descriptive summaries of the selected records. They are not medical diagnoses, treatment guidance, or causal conclusions.
