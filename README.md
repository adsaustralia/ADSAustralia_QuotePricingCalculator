
# Tender SQM Mapping Wizard – v13 (No Export Yet)

This version implements the new **concept flow**:

1. Upload Excel
2. Preview each sheet
3. Map columns in the UI:
   - Stock/Material
   - Size/Dimensions
   - Quantity (annual)
   - Double/Single sided (optional)
   - Runs per annum (optional)
   - Per-run qty (optional)
   - Lot ID / Description (optional)
4. Auto-generate material groups (with manual editing)
5. Control over Double Sided? at line level
6. Multi-panel area parsing (e.g. "2 x 2547mm x 755mm 2 x 967mm x 755mm")
7. Tiered pricing per m² by annual quantity (3 tiers)
8. Preview with:
   - Area m² per item
   - Total Area m²
   - Price per m² (tiered)
   - Double-sided loading
   - Line Value (ex GST)
   - Per-run metrics if runs are mapped

**Important:** Output/export is *not yet implemented* by design.
Once the required output format is confirmed (which sheet/columns/prices
need to be written), the export step can be added on top of this.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```
