# Blend4 Model Monitoring

Runbook to rebuild the Blend4 monitoring bases and generate the credit analysis (CR) and funnel reports.

## Prerequisites

- Python 3.10+ and Jupyter
- Access to the BigQuery project `loft-dl-fintech` (Application Default Credentials)
- Local packages used by the notebooks:

```bash
pip install pandas numpy matplotlib seaborn pandas-gbq google-cloud-bigquery google-auth unidecode requests nbformat nbconvert beautifulsoup4
```

Authenticate once before running the base notebooks:

```bash
gcloud auth application-default login
```

The notebooks create `data/raw`, `data/trusted`, and `data/analytics` if they do not exist. Intermediate CSVs stay local (`data/` is gitignored).

The `02` notebooks also expect these **one-time reference artifacts** already in `data/analytics/` (they are not rebuilt every run):

| File | Used by |
|------|---------|
| `blend4_psi_reference.pkl` | `02.Monitoramento_Blend4.ipynb` |
| `blend4_psi_baseline_ref.csv` | `02.Monitoramento_Blend4.ipynb` |
| `blend4_bvs_score_psi_reference.pkl` | `02.Monitoramento_Blend4.ipynb` |
| `psi_income_rental_reference.pkl` | `02.Monitoramento_Blend4.ipynb` |
| `dev_rating_pol_blend4.csv` | `02.Monitoramento_Funil_Blend4.ipynb` |

If they are missing, generate them once with `01.Monitoramento_Variaveis_PSI_Blend4.ipynb`.

## Run sequence

Run all notebooks **from the repository root**, in this order.

### 1. Build the bases

These two notebooks pull production data from BigQuery and write the CSVs used by the monitors.

| Order | Notebook | What it does | Output |
|------|----------|--------------|--------|
| 1 | `01.Base_Monitoramento_CR.ipynb` | Loads CredPago credit analyses, parses request/response JSON, scores Blend4, and builds ratings | `data/analytics/df_predict_blend4.csv` |
| 2 | `01.Base_Monitoramento_Funil.ipynb` | Loads the PF funnel from BigQuery and joins it with the CR base | `data/analytics/df_funil_blend4.csv` |

`01.Base_Monitoramento_Funil.ipynb` reads `df_predict_blend4.csv`, so it must run **after** the CR notebook.

### 2. Run the monitors

After both bases exist, run the two monitoring notebooks (order between them does not matter):

| Notebook | Input | What it covers |
|----------|-------|----------------|
| `02.Monitoramento_Blend4.ipynb` | `df_predict_blend4.csv` | Production Blend4 scores and ratings, rating mix vs Blend3, variable PSI (weekly and vs development), raw BVS / income / rent PSI, multi vs single proponent, rating cutoffs |
| `02.Monitoramento_Funil_Blend4.ipynb` | `df_funil_blend4.csv` | Daily/weekly funnel, conversion tables, model and rating mix, expected vs production policy mix, income follow-up, city and agency mix |

Each `02` notebook ends with a cell that exports an HTML report via `build_report_html.py`.

## Where to read the monitors

- **In the notebooks**: open `02.Monitoramento_Blend4.ipynb` or `02.Monitoramento_Funil_Blend4.ipynb` and review the charts and tables in place.
- **As HTML reports**: after the last cell of each `02` notebook, open:

  - `Monitores/02.Monitoramento_Blend4_report.html`
  - `Monitores/02.Monitoramento_Funil_Blend4_report.html`

Save the notebook (Cmd+S / Ctrl+S) before re-running the export cell so the HTML includes the latest outputs.

## Repository layout (main files)

```
01.Base_Monitoramento_CR.ipynb          # Step 1 — CR base
01.Base_Monitoramento_Funil.ipynb       # Step 2 — funnel base
02.Monitoramento_Blend4.ipynb           # Step 3 — CR monitor
02.Monitoramento_Funil_Blend4.ipynb     # Step 4 — funnel monitor
Monitores/                              # HTML reports
funcoes_escoragem.py                    # Blend4 scoring
funcoes_monitoramento.py                # Charts and tables
funcoes_psi.py                          # PSI helpers
build_report_html.py                    # Notebook → HTML export
data/analytics/                         # Local CSVs and PSI references (not in git)
```

`Codigo_BLEND3/` keeps the previous Blend3 pipeline and is not part of this run.
