
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Tender SQM Mapping Wizard v13.3", layout="wide")
st.title("Tender SQM Mapping Wizard – v13.3 (Auto-detect Row/Column + Fix)")

uploaded = st.file_uploader("Upload Excel", type=["xlsx","xls"])
if not uploaded:
    st.stop()

xls = pd.ExcelFile(uploaded)
sheet_name = st.selectbox("Choose sheet", xls.sheet_names)
df_raw = xls.parse(sheet_name)

st.markdown("### Raw Preview")
st.dataframe(df_raw.head(30), use_container_width=True)

rows, cols = df_raw.shape
if rows >= cols * 2:
    default_layout = "Row Based"
elif cols >= rows * 2:
    default_layout = "Column Based"
else:
    default_layout = "Row Based"

layout_type = st.radio(
    "Is this sheet Row-based or Column-based?",
    ["Row Based", "Column Based"],
    index=0 if default_layout == "Row Based" else 1,
    help=("Row-based = each tender item is a ROW.\n"
          "Column-based = each tender item is a COLUMN (the sheet will be transposed).")
)

if layout_type == "Row Based":
    df = df_raw.copy()
else:
    df = df_raw.T.copy()
    df.columns = [f"Col_{i}" for i in range(1, len(df.columns)+1)]

st.markdown("### Normalised View")
st.dataframe(df.head(30), use_container_width=True)

st.success("v13.3 base structure loaded. Full logic will be merged next.")
