import pandas as pd
import streamlit as st

FILE = "sample_pl.xlsx"

# Load your existing Excel workbook
df = pd.read_excel(FILE, sheet_name="P&L Detail")

# Ensure financial amounts are numeric
df["Amount USD"] = pd.to_numeric(
    df["Amount USD"], errors="coerce"
).fillna(0)
