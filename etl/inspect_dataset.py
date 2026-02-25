import os
import pandas as pd

# Build correct absolute path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/raw/online_retail_II.xlsx")

def inspect():
    print("Loading dataset...")

    # Load both sheets
    df_2009 = pd.read_excel(DATA_PATH, sheet_name=0)
    df_2010 = pd.read_excel(DATA_PATH, sheet_name=1)

    df = pd.concat([df_2009, df_2010], ignore_index=True)

    print("\n--- BASIC INFO ---")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\n--- DATE RANGE ---")
    print("Min date:", df["InvoiceDate"].min())
    print("Max date:", df["InvoiceDate"].max())

    print("\n--- UNIQUE CUSTOMERS ---")
    print("Unique CustomerIDs:", df["CustomerID"].nunique())

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

if __name__ == "__main__":
    inspect()