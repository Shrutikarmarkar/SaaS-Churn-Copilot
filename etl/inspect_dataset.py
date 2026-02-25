import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Put your xlsx locally here (ignored by git)
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "online_retail_II.xlsx")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep a stable canonical naming convention for downstream ETL.
    We convert:
      'Customer ID' -> 'CustomerID'
      and remove spaces in general.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df.columns = [c.replace(" ", "") for c in df.columns]  # Customer ID -> CustomerID
    return df

def inspect(data_path: str = DEFAULT_DATA_PATH) -> None:
    print("Using data path:", data_path)
    print("File exists:", os.path.exists(data_path))
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Put the xlsx inside data/raw/ as online_retail_II.xlsx"
        )

    # Load both sheets by name (matches what you see in Excel tabs)
    sheet_1 = "Year 2009-2010"
    sheet_2 = "Year 2010-2011"

    print("\nLoading sheets...")
    df1 = pd.read_excel(data_path, sheet_name=sheet_1)
    df2 = pd.read_excel(data_path, sheet_name=sheet_2)

    print("\n--- RAW COLUMNS (Sheet 1) ---")
    print(df1.columns.tolist())
    print("\n--- RAW COLUMNS (Sheet 2) ---")
    print(df2.columns.tolist())

    # Combine + normalize column names for consistent pipeline
    df = pd.concat([df1, df2], ignore_index=True)
    df = normalize_columns(df)

    # Parse InvoiceDate
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    print("\n==============================")
    print("COMBINED DATASET INSPECTION")
    print("==============================")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\n--- SAMPLE ROWS ---")
    print(df.head(5).to_string(index=False))

    print("\n--- DTYPES ---")
    print(df.dtypes)

    print("\n--- DATE RANGE (InvoiceDate) ---")
    print("Min:", df["InvoiceDate"].min())
    print("Max:", df["InvoiceDate"].max())
    print("Null InvoiceDate:", df["InvoiceDate"].isna().sum())

    print("\n--- UNIQUES ---")
    print("Unique Invoice:", df["Invoice"].nunique() if "Invoice" in df.columns else "NA")
    print("Unique CustomerID:", df["CustomerID"].nunique() if "CustomerID" in df.columns else "NA")
    print("Unique StockCode:", df["StockCode"].nunique() if "StockCode" in df.columns else "NA")
    print("Unique Country:", df["Country"].nunique() if "Country" in df.columns else "NA")

    print("\n--- MISSING VALUES (Top) ---")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0].head(20))

    print("\n--- BASIC SANITY CHECKS ---")
    if "Quantity" in df.columns:
        print("Quantity < 0 count:", (df["Quantity"] < 0).sum())
        print("Quantity == 0 count:", (df["Quantity"] == 0).sum())
    if "Price" in df.columns:
        print("Price <= 0 count:", (df["Price"] <= 0).sum())

    print("\nDone.")

if __name__ == "__main__":
    inspect()