from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent /"data" / "raw" / "online_retail_II.csv"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
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
    is_cancel = df["Invoice"].str.startswith("C")
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

def add_columns(df):
    """Shton kolonat e reja: te ardhurat dhe kolonat e kohes."""
    
    df = df.copy()
    df["Customer ID"] = df["Customer ID"].astype("Int64")
    df["Revenue"] = df["Quantity"] * df["Price"]
    df["Date"] = df["InvoiceDate"].to_normalize()
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour
    return df

def validate_data(sales, returns):
    """Kontrollon qe pastrimi ka punuar. Ndalon me gabim nese diqka nuk eshte ne rregull"""
    
    assert (sales["Quantity"] > 0).all(), "sales ka Quantity <= 0"
    assert (sales["Price"] > 0).all(), "sales ka Price <= 0"
    assert (sales["Revenue"] > 0).all(), "sales ka Revenue <= 0"
    assert not sales["Invoice"].str.startswith("C").any(), "sales ka anulime"
    assert returns["Invoice"].str.startswith("C").all(), "returns ka fatura pa C"
    assert not sales.duplicated().any(), "sales ka dublikate"
    assert sales["StockCode"].str.match(PRODUCT_CODE_PATTERN).all(), "sales ka kode te cilat nuk jane produkte"
    assert sales["InvoiceDate"].notna().all(), "sales ka data qe mungojne"
    print("Kontrollet e cilesise: OK")
    
def save_processed(sales, returns, folder=PROCESSED_DIR):
    """Ruan te dhenat e pastra te data/processed/."""
    
    folder.mkdir(parents=True, exist_ok=True)
    sales.to_csv(folder / "sales_clean.csv", index=False)
    returns.to_csv(folder / "returns.csv", index=False)
    print(f"U ruajten {len(sales):,} rreshta shitjesh dhe {len(returns):,} rreshta kthimesh te {folder}")
    
