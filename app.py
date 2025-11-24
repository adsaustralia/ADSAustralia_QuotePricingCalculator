
import io
import re
import numpy as np
import pandas as pd
import streamlit as st


def parse_area_m2(dimensions: str):
    if pd.isna(dimensions):
        return np.nan
    s = str(dimensions).lower().replace("×", "x")
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
    if not isinstance(stock, str):
        return ""
    s_raw = stock
    s = stock.lower()
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
    if "sav" in s or "vinyl" in s:
        return "SAV / Vinyl"
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


def num_to_col(n: int) -> str:
    res = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        res = chr(ord("A") + rem) + res
    return res


st.set_page_config(page_title="Tender SQM Mapping Wizard v13.5", layout="wide")
st.title("Tender SQM Mapping Wizard – v13.5 (Excel-style views)")


uploaded = st.file_uploader("1. Upload Excel file", type=["xlsx", "xls"])
if not uploaded:
    st.stop()

xls = pd.ExcelFile(uploaded)
sheet_name = st.selectbox("2. Choose sheet to map", xls.sheet_names)
df_raw = xls.parse(sheet_name)

st.markdown("#### Preview of raw sheet (Excel-style)")
raw_rows = len(df_raw)
raw_cols = df_raw.shape[1]
raw_preview = df_raw.copy()
raw_preview.index = range(1, raw_rows + 1)
raw_preview.columns = [num_to_col(i + 1) for i in range(raw_cols)]

with st.expander("Raw sheet view controls", expanded=True):
    raw_start_row = st.number_input("Start row", min_value=1, max_value=max(1, raw_rows), value=1)
    raw_end_row = st.number_input("End row", min_value=raw_start_row, max_value=raw_rows, value=min(raw_rows, raw_start_row + 49))
    raw_visible_cols = st.multiselect("Columns to display", options=list(raw_preview.columns), default=list(raw_preview.columns))

raw_display = raw_preview.loc[raw_start_row:raw_end_row, raw_visible_cols]
st.dataframe(raw_display, use_container_width=True)

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
        "Row-based = each tender item is a ROW.\n"
        "Column-based = each tender item is a COLUMN (the sheet will be transposed for mapping)."
    ),
)

if layout_type == "Row Based":
    df = df_raw.copy()
else:
    df = df_raw.T.copy()

st.markdown("#### 4. Normalised view used for mapping (Excel-style)")
norm_rows = len(df)
norm_cols = df.shape[1]
norm_preview = df.copy()
norm_preview.index = range(1, norm_rows + 1)
norm_letters = [num_to_col(i + 1) for i in range(norm_cols)]
norm_preview.columns = norm_letters

with st.expander("Normalised view controls", expanded=True):
    norm_start_row = st.number_input("Normalised start row", min_value=1, max_value=max(1, norm_rows), value=1, key="norm_start_row")
    norm_end_row = st.number_input("Normalised end row", min_value=norm_start_row, max_value=norm_rows, value=min(norm_rows, norm_start_row + 49), key="norm_end_row")
    norm_visible_cols = st.multiselect("Normalised columns to display", options=list(norm_preview.columns), default=list(norm_preview.columns), key="norm_visible_cols")

norm_display = norm_preview.loc[norm_start_row:norm_end_row, norm_visible_cols]
st.dataframe(norm_display, use_container_width=True)

df_cols = list(df.columns)
letter_to_col = dict(zip(norm_letters, df_cols))


def choice_to_col(choice: str):
    if "–" in choice:
        letter = choice.split("–", 1)[0].strip()
    elif "-" in choice:
        letter = choice.split("-", 1)[0].strip()
    else:
        letter = choice.strip()
    return letter_to_col.get(letter)


col_labels = [f"{ltr} – {str(col)}" for ltr, col in zip(norm_letters, df_cols)]

st.markdown("### 5. Map columns to fields (by Excel letter)")
col1, col2 = st.columns(2)
with col1:
    material_choice = st.selectbox("Stock / Material column", col_labels)
    size_choice = st.selectbox("Size / Dimensions column", col_labels)
    qty_choice = st.selectbox("Quantity (annual) column", col_labels)
with col2:
    sides_choice = st.selectbox("Double/Single sided column (optional)", ["<none>"] + col_labels)
    runs_choice = st.selectbox("Runs per annum column (optional)", ["<none>"] + col_labels)
    per_run_choice = st.selectbox("Per-run qty column (optional)", ["<none>"] + col_labels)

col3, col4 = st.columns(2)
with col3:
    lot_choice = st.selectbox("Lot ID column (optional)", ["<none>"] + col_labels)
with col4:
    desc_choice = st.selectbox("Description column (optional)", ["<none>"] + col_labels)

if not st.button("6. Apply mapping and continue"):
    st.stop()

material_col = choice_to_col(material_choice)
size_col = choice_to_col(size_choice)
qty_col = choice_to_col(qty_choice)
sides_col = choice_to_col(sides_choice) if sides_choice != "<none>" else None
runs_col = choice_to_col(runs_choice) if runs_choice != "<none>" else None
per_run_col = choice_to_col(per_run_choice) if per_run_choice != "<none>" else None
lot_col = choice_to_col(lot_choice) if lot_choice != "<none>" else None
desc_col = choice_to_col(desc_choice) if desc_choice != "<none>" else None

data = pd.DataFrame()
data["Stock / Material"] = df[material_col]
data["Dimensions"] = df[size_col]
data["Quantity"] = pd.to_numeric(df[qty_col], errors="coerce")

if lot_col is not None:
    data["Lot ID"] = df[lot_col]
if desc_col is not None:
    data["Description"] = df[desc_col]

if runs_col is not None:
    data["Runs per Annum"] = pd.to_numeric(df[runs_col], errors="coerce")
else:
    data["Runs per Annum"] = np.nan

if per_run_col is not None:
    data["Per-run Qty"] = pd.to_numeric(df[per_run_col], errors="coerce")
else:
    data["Per-run Qty"] = np.nan

if sides_col is not None:
    raw_side = df[sides_col]
    data["Sided (auto)"] = raw_side.apply(detect_sides_from_text)
else:
    data["Sided (auto)"] = data["Stock / Material"].apply(detect_sides_from_text)

data["Double Sided?"] = data["Sided (auto)"].apply(lambda s: s == "Double Sided")
data["Area m² (each)"] = data["Dimensions"].apply(parse_area_m2)
data["Total Area m²"] = data["Area m² (each)"] * data["Quantity"]

if runs_col is not None:
    safe_runs = data["Runs per Annum"].replace(0, np.nan)
    data["Area m² per Run"] = data["Total Area m²"] / safe_runs
else:
    data["Area m² per Run"] = np.nan

st.markdown("### 7. Grouping & double-sided control")
unique_materials = sorted([s for s in data["Stock / Material"].dropna().unique() if str(s).strip()])

if "groups_df" not in st.session_state:
    st.session_state["groups_df"] = pd.DataFrame(
        {"Stock / Material": unique_materials,
         "Initial Group": [material_group_key(s) for s in unique_materials]}
    )
    st.session_state["groups_df"]["Assigned Group"] = st.session_state["groups_df"]["Initial Group"]
else:
    gdf = st.session_state["groups_df"]
    existing = set(gdf["Stock / Material"])
    new_mats = [s for s in unique_materials if s not in existing]
    if new_mats:
        new_rows = pd.DataFrame(
            {"Stock / Material": new_mats,
             "Initial Group": [material_group_key(s) for s in new_mats]}
        )
        new_rows["Assigned Group"] = new_rows["Initial Group"]
        gdf = pd.concat([gdf, new_rows], ignore_index=True)
    gdf = gdf[gdf["Stock / Material"].isin(unique_materials)].reset_index(drop=True)
    st.session_state["groups_df"] = gdf

groups_df = st.session_state["groups_df"]

st.markdown(
    "- **Initial Group** is auto-generated based on material text (thickness, GSM, SAV, etc.).  
"
    "- **Assigned Group** is what actually drives pricing.  
"
    "- Change Assigned Group values to merge or split stock groups."
)

groups_df = st.data_editor(
    groups_df,
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "Stock / Material": st.column_config.TextColumn(disabled=True),
        "Initial Group": st.column_config.TextColumn(disabled=True),
    },
    key="group_editor",
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
    key="sided_editor",
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

double_loading_pct = st.sidebar.number_input("Double-sided loading (%)", min_value=0.0, value=25.0, step=1.0)
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
data["Line Value (ex GST)"] = data["Total Area m²"] * data["Price per m²"] * data["Sided Multiplier"]

if runs_col is not None:
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
if runs_col is not None:
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

st.info("Export is not implemented yet. Once you confirm the output structure (which price columns where), it can be added.")
