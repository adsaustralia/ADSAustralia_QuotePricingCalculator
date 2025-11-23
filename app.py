
import io
import re
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Helper functions
# -----------------------------


def parse_area_m2(dimensions: str):
    """Convert dimension text to m².

    Handles:
    - '841mm x 1189mm'
    - '2 x 2547mm x 755mm 2 x 967mm x 755mm' (multi-panel)
    """
    if pd.isna(dimensions):
        return np.nan

    s = str(dimensions).lower().replace("×", "x")

    # Multi-panel pattern: "2 x 2547mm x 755mm"
    panel_pattern = r"(\d+)\s*x\s*(\d+)\s*mm\s*x\s*(\d+)\s*mm"
    matches = list(re.finditer(panel_pattern, s))
    if matches:
        total_mm2 = 0.0
        for m in matches:
            qty = int(m.group(1))
            w = float(m.group(2))
            h = float(m.group(3))
            total_mm2 += qty * w * h
        return total_mm2 / 1_000_000.0

    # Fallback: simple "841mm x 1189mm" style
    simple = s.replace(" ", "").replace("mm", "")
    parts = simple.split("x")
    if len(parts) != 2:
        return np.nan
    try:
        w = float(parts[0])
        h = float(parts[1])
    except Exception:
        return np.nan
    return (w * h) / 1_000_000.0


def detect_sides_from_text(text: str):
    if pd.isna(text):
        return "Single Sided"
    s = str(text).lower()
    if "double" in s or "ds" in s:
        return "Double Sided"
    if "single" in s or "ss" in s:
        return "Single Sided"
    return "Single Sided"


def material_group_key(stock: str) -> str:
    """Medium-level grouping based on stock/material text."""
    if not isinstance(stock, str):
        return ""
    s_raw = stock
    s = stock.lower()

    # Thickness based grouping
    m_thick = re.search(r"(\d+)\s*mm", s)
    if m_thick:
        thickness = m_thick.group(1)
        if "screenboard" in s or "screen board" in s:
            return f"{thickness}mm Screenboard"
        if "corflute" in s or "coreflute" in s:
            return f"{thickness}mm Corflute"
        if "acrylic" in s:
            return f"{thickness}mm Acrylic"
        if "pvc" in s:
            return f"{thickness}mm PVC"
        if "hips" in s:
            return f"{thickness}mm HIPS"
        if "acm" in s:
            return f"{thickness}mm ACM"

    # GSM based
    m_gsm = re.search(r"(\d{3})\s*gsm", s)
    if m_gsm:
        gsm = m_gsm.group(1)
        if "silk" in s or "satin" in s:
            return f"{gsm}gsm Silk/Satin"
        if "matt" in s or "ecomatt" in s:
            return f"{gsm}gsm Matt"
        if "gloss" in s:
            return f"{gsm}gsm Gloss"
        if "synthetic" in s or "plasnet" in s:
            return f"{gsm}gsm Synthetic"
        return f"{gsm}gsm Paper/Card"

    # Simple SAV grouping
    if "sav" in s or "vinyl" in s:
        return "SAV / Vinyl"

    # Fallback: first two "words"
    cleaned = re.sub(r"\(.*?\)", "", s)
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned).strip()
    tokens = cleaned.split()
    if len(tokens) >= 2:
        return " ".join(tokens[:2])
    elif tokens:
        return tokens[0]
    return s_raw.strip()


def fmt_money(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return ""
    return f"${x:,.2f}"


def get_tiered_rate(qty, tiers):
    """
    tiers: list of dicts with keys: min_qty, max_qty, price
    qty: quantity (annual)

    Returns price per m² based on where qty falls.
    """
    if pd.isna(qty):
        return 0.0
    for t in tiers:
        if t["min_qty"] is None and t["max_qty"] is None:
            continue
        if t["min_qty"] is None and qty <= t["max_qty"]:
            return t["price"]
        if t["max_qty"] is None and qty >= t["min_qty"]:
            return t["price"]
        if t["min_qty"] is not None and t["max_qty"] is not None:
            if t["min_qty"] <= qty <= t["max_qty"]:
                return t["price"]
    return 0.0


# -----------------------------
# Streamlit app
# -----------------------------


st.set_page_config(page_title="Tender SQM Mapping Wizard v13.2", layout="wide")
st.title("Tender SQM Mapping Wizard – v13.2 (Row/Column Auto-Detect)")


uploaded = st.file_uploader("1. Upload Excel file", type=["xlsx", "xls"])
if not uploaded:
    st.stop()

xls = pd.ExcelFile(uploaded)

sheet_name = st.selectbox("2. Choose sheet to map", xls.sheet_names)
df_raw = xls.parse(sheet_name)

st.markdown("#### Preview of raw sheet (first 30 rows)")
st.dataframe(df_raw.head(30), use_container_width=True)

# Auto-detect layout type based on shape
rows, cols = df_raw.shape
if rows >= cols * 2:
    default_layout = "Row Based"
elif cols >= rows * 2:
    default_layout = "Column Based"
else:
    default_layout = "Row Based"

layout_type = st.radio(
    "3. Is this sheet Row-based or Column-based?",
    ["Row Based", "Column Based"],
    index=0 if default_layout == "Row Based" else 1,
    help=(
        "Row-based = each tender item is a ROW.
"
        "Column-based = each tender item is a COLUMN (the sheet will be transposed for mapping)."
    ),
)

# Normalise structure for mapping
if layout_type == "Row Based":
    df = df_raw.copy()
else:
    df = df_raw.T.copy()
    # Give generic column names after transpose
    df.columns = [f"Col_{i}" for i in range(1, len(df.columns) + 1)]

st.markdown("#### 4. Normalised view used for mapping (first 30 rows)")
st.dataframe(df.head(30), use_container_width=True)

cols = df.columns.tolist()

st.markdown("### 5. Map columns to fields")

col1, col2 = st.columns(2)
with col1:
    material_col = st.selectbox("Stock / Material column", cols)
    size_col = st.selectbox("Size / Dimensions column", cols)
    qty_col = st.selectbox("Quantity (annual) column", cols)
with col2:
    sides_col = st.selectbox("Double/Single sided column (optional)", ["<none>"] + cols)
    runs_col = st.selectbox("Runs per annum column (optional)", ["<none>"] + cols)
    per_run_col = st.selectbox("Per-run qty column (optional)", ["<none>"] + cols)

col3, col4 = st.columns(2)
with col3:
    lot_col = st.selectbox("Lot ID column (optional)", ["<none>"] + cols)
with col4:
    desc_col = st.selectbox("Description column (optional)", ["<none>"] + cols)

if not st.button("6. Apply mapping and continue"):
    st.stop()

# -----------------------------
# Build normalized data table
# -----------------------------

data = pd.DataFrame()
data["Stock / Material"] = df[material_col]
data["Dimensions"] = df[size_col]
data["Quantity"] = pd.to_numeric(df[qty_col], errors="coerce")

if lot_col != "<none>":
    data["Lot ID"] = df[lot_col]
if desc_col != "<none>":
    data["Description"] = df[desc_col]

if runs_col != "<none>":
    data["Runs per Annum"] = pd.to_numeric(df[runs_col], errors="coerce")
else:
    data["Runs per Annum"] = np.nan

if per_run_col != "<none>":
    data["Per-run Qty"] = pd.to_numeric(df[per_run_col], errors="coerce")
else:
    data["Per-run Qty"] = np.nan

# Sidedness
if sides_col != "<none>":
    raw_side = df[sides_col]
    data["Sided (auto)"] = raw_side.apply(detect_sides_from_text)
else:
    data["Sided (auto)"] = data["Stock / Material"].apply(detect_sides_from_text)

data["Double Sided?"] = data["Sided (auto)"].apply(lambda s: s == "Double Sided")

# Area / per-run calcs
data["Area m² (each)"] = data["Dimensions"].apply(parse_area_m2)
data["Total Area m²"] = data["Area m² (each)"] * data["Quantity"]

if runs_col != "<none>":
    safe_runs = data["Runs per Annum"].replace(0, np.nan)
    data["Area m² per Run"] = data["Total Area m²"] / safe_runs
else:
    data["Area m² per Run"] = np.nan

st.markdown("### 7. Grouping & double-sided control")

unique_materials = sorted([s for s in data["Stock / Material"].dropna().unique() if str(s).strip()])

if "groups_df" not in st.session_state:
    st.session_state["groups_df"] = pd.DataFrame({
        "Stock / Material": unique_materials,
        "Initial Group": [material_group_key(s) for s in unique_materials]
    })
    st.session_state["groups_df"]["Assigned Group"] = st.session_state["groups_df"]["Initial Group"]
else:
    gdf = st.session_state["groups_df"]
    existing = set(gdf["Stock / Material"])
    new_mats = [s for s in unique_materials if s not in existing]
    if new_mats:
        new_rows = pd.DataFrame({
            "Stock / Material": new_mats,
            "Initial Group": [material_group_key(s) for s in new_mats]
        })
        new_rows["Assigned Group"] = new_rows["Initial Group"]
        gdf = pd.concat([gdf, new_rows], ignore_index=True)
    gdf = gdf[gdf["Stock / Material"].isin(unique_materials)].reset_index(drop=True)
    st.session_state["groups_df"] = gdf

groups_df = st.session_state["groups_df"]

st.markdown(
    """
- **Initial Group** is auto-generated based on material text (thickness, GSM, SAV, etc.).  
- **Assigned Group** is what actually drives pricing.  
- Change Assigned Group values to merge or split stock groups.
    """
)

groups_df = st.data_editor(
    groups_df,
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "Stock / Material": st.column_config.TextColumn(disabled=True),
        "Initial Group": st.column_config.TextColumn(disabled=True),
    },
    key="group_editor"
)
st.session_state["groups_df"] = groups_df

mat_to_group = dict(zip(groups_df["Stock / Material"], groups_df["Assigned Group"]))
data["Material Group"] = data["Stock / Material"].map(mat_to_group).fillna("Unassigned")

st.markdown("#### Edit double-sided flags per line")

editable_cols = ["Stock / Material", "Dimensions", "Quantity", "Material Group", "Double Sided?"]
if "Lot ID" in data.columns:
    editable_cols.insert(0, "Lot ID")
if "Description" in data.columns:
    editable_cols.insert(1, "Description")

edited_lines = st.data_editor(
    data[editable_cols],
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "Double Sided?": st.column_config.CheckboxColumn("Double Sided?")
    },
    key="sided_editor"
)

data["Double Sided?"] = edited_lines["Double Sided?"].fillna(False)

st.markdown("### 8. Preview merged groups & sidedness")

group_summary = (
    data.groupby("Material Group")
    .agg(
        Materials=("Stock / Material", "nunique"),
        Lines=("Stock / Material", "count"),
        Total_Area_m2=("Total Area m²", "sum"),
        Double_Sided_Lines=("Double Sided?", "sum"),
    )
    .reset_index()
)

group_summary["Single_Sided_Lines"] = group_summary["Lines"] - group_summary["Double_Sided_Lines"]
group_summary["Total_Area_m2"] = group_summary["Total_Area_m2"].round(2)

st.dataframe(group_summary, use_container_width=True)

st.markdown("### 9. Pricing (with tiers) & preview")

st.sidebar.header("Pricing controls")

double_loading_pct = st.sidebar.number_input(
    "Double-sided loading (%)",
    min_value=0.0,
    value=25.0,
    step=1.0,
)

st.sidebar.subheader("Tiered price per m² (by annual quantity)")

tier1_max = st.sidebar.number_input("Tier 1: max qty", min_value=1, value=100, step=1)
tier1_price = st.sidebar.number_input("Tier 1: price $/m²", min_value=0.0, value=10.0, step=0.1)

tier2_max = st.sidebar.number_input("Tier 2: max qty", min_value=tier1_max, value=1000, step=1)
tier2_price = st.sidebar.number_input("Tier 2: price $/m²", min_value=0.0, value=8.0, step=0.1)

tier3_price = st.sidebar.number_input("Tier 3: price $/m² (qty > Tier 2)", min_value=0.0, value=6.0, step=0.1)

tiers = [
    {"min_qty": None, "max_qty": tier1_max, "price": tier1_price},
    {"min_qty": tier1_max + 1, "max_qty": tier2_max, "price": tier2_price},
    {"min_qty": tier2_max + 1, "max_qty": None, "price": tier3_price},
]

data["Price per m²"] = data["Quantity"].apply(lambda q: get_tiered_rate(q, tiers))

double_mult = 1.0 + double_loading_pct / 100.0
data["Sided Multiplier"] = np.where(data["Double Sided?"], double_mult, 1.0)

data["Line Value (ex GST)"] = (
    data["Total Area m²"] * data["Price per m²"] * data["Sided Multiplier"]
)

if runs_col != "<none>":
    safe_runs_val = data["Runs per Annum"].replace(0, np.nan)
    data["Value per Run (ex GST)"] = data["Line Value (ex GST)"] / safe_runs_val
else:
    data["Value per Run (ex GST)"] = np.nan

preview_cols = [
    "Stock / Material",
    "Dimensions",
    "Quantity",
    "Material Group",
    "Double Sided?",
    "Area m² (each)",
    "Total Area m²",
    "Price per m²",
    "Sided Multiplier",
    "Line Value (ex GST)",
]
if runs_col != "<none>":
    preview_cols.insert(preview_cols.index("Total Area m²") + 1, "Runs per Annum")
    preview_cols.insert(preview_cols.index("Runs per Annum") + 1, "Area m² per Run")
    preview_cols.insert(preview_cols.index("Line Value (ex GST)"), "Value per Run (ex GST)")

if "Lot ID" in data.columns:
    preview_cols.insert(0, "Lot ID")
if "Description" in data.columns:
    preview_cols.insert(1, "Description")

display_data = data[preview_cols].copy()
display_data["Area m² (each)"] = display_data["Area m² (each)"].round(3)
display_data["Total Area m²"] = display_data["Total Area m²"].round(2)
if "Area m² per Run" in display_data.columns:
    display_data["Area m² per Run"] = display_data["Area m² per Run"].round(2)
display_data["Price per m²"] = display_data["Price per m²"].apply(fmt_money)
display_data["Line Value (ex GST)"] = display_data["Line Value (ex GST)"].apply(fmt_money)
if "Value per Run (ex GST)" in display_data.columns:
    display_data["Value per Run (ex GST)"] = display_data["Value per Run (ex GST)"].apply(fmt_money)

st.dataframe(display_data, use_container_width=True)

total_area = data["Total Area m²"].sum(skipna=True)
total_value = data["Line Value (ex GST)"].sum(skipna=True)

k1, k2 = st.columns(2)
with k1:
    st.metric("Total Area (m² per annum)", f"{total_area:,.2f}")
with k2:
    st.metric("Total Value (ex GST)", fmt_money(total_value))

st.info("Export/download not implemented yet. Once you define the exact output format, we can add the Excel export step.")
