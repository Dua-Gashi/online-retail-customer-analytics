import numpy as np
import pandas as pd

def build_rfm(sales, snapshot_date=None):
    """Nderton tabelen RFM: nje rresht per klient.
    
    Recency   = dite nga blerja e fundit deri te data e references
    Frequency = numri i faturave unike
    Monetary  = shuma e te ardhurave     
    """
    customers = sales.dropna(subset=["Customer ID"])
    
    if snapshot_date is None:
        snapshot_date = customers["Date"].max() + pd.Timedelta(days=1)
    print(f"Data e references: {snapshot_date.date()}")
    
    rfm = customers.groupby("Customer ID").agg(
        LastPurchase=("Date", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum")
    )
    rfm["Recency"] = (snapshot_date - rfm["LastPurchase"]).dt.days
    
    return rfm[["Recency", "Frequency", "Monetary"]]

def log_transform(rfm):
    """Zbaton log(1 + x) mbi R, F, M, sepse te tria jane shume te anuara djathtas."""
    return np.log1p(rfm[["Recency", "Frequency", "Monetary"]]).add_suffix("_log")

def validate_rfm(rfm):
    """Kontrollon qe tabela RFM eshte e sakte."""
    
    assert rfm.index.is_unique, "Customer ID i perseritur"
    assert not rfm.isna().any().any(), "RFM ka vlera qe mungojne"
    assert (rfm["Recency"] >= 1).all(), "Requency < 1"
    assert (rfm["Frequency"] >= 1).all(), "Frequency < 1"
    assert (rfm["Monetary"] > 0).all(), "Monetary <= 0"
    print("Kontrollet RFM: OK")    