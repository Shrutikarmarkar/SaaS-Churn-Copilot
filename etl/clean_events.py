import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "online_retail_II.xlsx")

OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
OUT_PATH = os.path.join(OUT_DIR, "events_clean.parquet")
SAMPLE_PATH = os.path.join(BASE_DIR, "data", "sample", "events_clean_sample.csv")

SHEET_1 = "Year 2009-2010"
SHEET_2 = "Year 2010-2011"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df.columns = [c.replace(" ", "") for c in df.columns]  # Customer ID -> CustomerID
    return df

def load_raw() -> pd.DataFrame:
    df1 = pd.read_excel(RAW_PATH, sheet_name=SHEET_1)
    df2 = pd.read_excel(RAW_PATH, sheet_name=SHEET_2)
    df = pd.concat([df1, df2], ignore_index=True)
    df = normalize_columns(df)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows without a CustomerID (cannot map to user/account)
    df = df[df["CustomerID"].notna()]

    # Cast CustomerID to int (it currently shows as float like 13085.0)
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Remove cancellations/returns:
    # 1) negative quantity
    df = df[df["Quantity"] > 0]

    # 2) cancellations invoices start with "C"
    df = df[~df["Invoice"].astype(str).str.startswith("C")]

    # Remove invalid price rows
    df = df[df["Price"] > 0]

    # Drop rows with missing dates (should be none)
    df = df[df["InvoiceDate"].notna()]

    # Add event_day for daily aggregation
    df["event_day"] = df["InvoiceDate"].dt.date
    df["event_day"] = pd.to_datetime(df["event_day"], errors="coerce")

    # Revenue proxy (for optional billing simulation)
    df["line_revenue"] = df["Quantity"] * df["Price"]

    # Force string columns (prevents pyarrow type inference issues)
    for col in ["Invoice", "StockCode", "Description", "Country"]:
        df[col] = df[col].astype(str)

    # Keep only columns we care about going forward
    keep = [
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "event_day", "Price", "line_revenue",
        "CustomerID", "Country"
    ]
    df = df[keep]

    return df

def save_outputs(df_clean: pd.DataFrame) -> None:
    print("Creating directories...")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data", "sample"), exist_ok=True)
    print("Directories created.")

    # Save full clean data locally (not committed to git ideally)
    df_clean.to_parquet(OUT_PATH, index=False)
    print(f"Saved cleaned events to: {OUT_PATH}")

    # Save a sample that CAN be committed to GitHub
    sample = df_clean.sample(n=min(50000, len(df_clean)), random_state=42)
    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"Saved sample to: {SAMPLE_PATH}")

def main():
    print("Loading raw...")
    df = load_raw()
    print("Raw shape:", df.shape)

    print("Cleaning...")
    df_clean = clean(df)
    print("Clean shape:", df_clean.shape)

    print("\nQuick checks:")
    print("Unique customers:", df_clean["CustomerID"].nunique())
    print("Date range:", df_clean["InvoiceDate"].min(), "→", df_clean["InvoiceDate"].max())
    print("Any negative quantity left?", (df_clean["Quantity"] <= 0).any())
    print("Any non-positive price left?", (df_clean["Price"] <= 0).any())

    save_outputs(df_clean)

if __name__ == "__main__":
    main()