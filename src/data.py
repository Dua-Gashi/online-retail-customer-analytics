from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent /"data" / "raw" / "online_retail_II.csv"
PRODUCT_CODE_PATTERN = r"^\d{5}"

def load_data(path=RAW_PATH):
    """Ngarko te dhenat e perpunuar nga CSV"""
    df = pd.read_csv(path, parse_dates=["Invoice"], low_memory=False)
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    return df

def _log(step, removed, remaining):
    print(f"{step:<42} hequr: {removed:>8} | mbeten: {remaining:>9,}")
    
def clean_data(df):
    """Pastron te dhenat dhe kthen (sales, returns).
    
    Rreshtat pa Customer ID mbeten qellimisht: jane te vlefshem per shitjet totale dhe hiqen vetem ne analizen e klienteve
    """
    df = df.copy()
    print(f"Rreshta fillestare: {len(df):,}")
    
    # 1. Dublikatet
    n = len(df)
    df = df.drop_duplicates()
    _log("Dublikate: ", n - len(df), len(df))
    
    # 2. Anulimet (faturat me C) ndahen veqmas
    is_cancel = df["IOnvoice"].str.startswith("C")
    returns = df[is_cancel].copy()
    df = df[~is_cancel]
    _log("Anulime (ndahen te returns)", len(returns), len(df))
    
    # 3. Sasi jo pozitive
    n = len(df)
    df = df[df["Quantity"] > 0]
    _log("Quantity <= 0", n - len(df), len(df))
    
    # 4. Cmime jo pozitive
    n = len(df)
    df  = df[df["Price"] > 0]
    _log("Price <= 0", n - len(df), len(df))
    
    # 5. Kode qe nuk jane produkte
    n = len(df)
    is_product = df["StockCode"].str.match(PRODUCT_CODE_PATTERN)
    removed_codes = df.loc[~is_product, "StockCode"].value_counts()
    df = df[is_product]
    _log("Kode te cilat nuk jane produkte", n-len(df), len(df))
    print("Kodet e hequra (top 10): ", removed_codes.head(10).to_dict())
    
    sales = df.reset_index(drop=True)
    print(f"Rreshta te shitjes pas pastrimit: {len(sales):,}")
    return sales, returns