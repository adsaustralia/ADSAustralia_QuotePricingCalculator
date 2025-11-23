
# Tender SQM Mapping Wizard – v13.2 (Row/Column Auto-Detect, No Export Yet)

This version adds:

- **Row vs Column layout auto-detection** per sheet, based on rows/columns count
- A radio control so the user can **override the guess** (Row Based / Column Based)
- For Column-based sheets, the app automatically **transposes** the data and assigns generic column names for mapping
- Then the existing flow runs:
  1. Sheet preview (raw + normalised)
  2. Column mapping for:
     - Stock/Material
     - Dimensions
     - Quantity (annual)
     - Double/Single sided (optional)
     - Runs per annum (optional)
     - Per-run qty (optional)
     - Lot ID / Description (optional)
  3. Auto-generated material groups + manual editing
  4. Double-sided line overrides
  5. Group summary preview
  6. Tiered pricing by annual quantity (3 tiers)
  7. Per-annum and per-run metrics, with money formatting

**Important:** Export/download of Excel is still intentionally not implemented.
Once the required output format is confirmed, a new version can be created with the export wired in.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```
