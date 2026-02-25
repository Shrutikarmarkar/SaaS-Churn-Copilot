import os
import requests

DATA_DIR = "data/raw"
FILE_PATH = os.path.join(DATA_DIR, "OnlineRetail.xlsx")

def download_dataset():
    print("Please manually download the Online Retail II dataset from:")
    print("https://archive.ics.uci.edu/ml/datasets/online+retail+ii")
    print(f"Place it inside: {FILE_PATH}")

if __name__ == "__main__":
    download_dataset()
