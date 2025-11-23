# ADS Tender SQM Calculator v12.7 (Mapping Wizard)

This version adds a **mapping wizard** so you can use almost any tender layout:

- Choose the sheet inside the workbook
- Map:
  - Dimensions (mm)
  - Total Annual Volume
  - Print/Stock Specifications
  - Lot ID (optional)
  - Item Description (optional)
  - Runs per annum (optional)
- The app then normalizes the data and runs the full pricing engine.

Features:

- ADS orange + navy theme
- ADS logo on the top-left (`ads_logo.png`)
- Option B material grouping
- Group + stock price memory (`price_memory.json`)
- Double-sided loading logic (configurable %)
- Multi-panel dimension parsing (e.g. "2 x 2547mm x 755mm 2 x 967mm x 755mm")
- Per-annum and per-run calculations:
  - Area m² per item
  - Total Area m² per annum
  - Area m² per run
  - Line Value (ex GST) per annum
  - Value per Run (ex GST)
- KPI cards:
  - Total Area (m² per annum)
  - Total Value (ex GST)
  - Average m² per Run
  - Average Value per Run (ex GST)
- Group preview with price and total value
- Final calculated table and Excel export

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```
