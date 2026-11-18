import pandas as pd

def build_daily_series(sales, drop_last_day=False):
    """Seria ditore e shitjeve: nje rresht per qdo dite kalendarike.
    
    revenue = te ardhurat e dites
    orders = numri i faturave unike
    is_open = True ne ka pasur te pakten nje fature
    Ditet pa asnje fature marrin revenue = 0 dhe orders = 0.
    """
    
    daily = sales.groupby("Date").agg(
        revenue=("Revenue", "sum"),
        orders=("Invoice", "nunique")
    )
    if drop_last_day:
        daily = daily.iloc[:-1]
        
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D", name="Date")
    daily = daily.reindex(full_index)
    
    daily["is_open"] = daily["orders"].notna()
    daily[["revenue", "orders"]] = daily[["revenue", "orders"]].fillna(0)
    daily["orders"] = daily["orders"].astype(int)
    return daily

def validate_daily(daily):
    """Kontrollon qe seria ditore eshte e plote dhe e sakt"""
    
    assert daily.index.is_unique and daily.index.is_monotonic_increasing, "Indeksi nuk eshte unik ose i renditur"
    steps = daily.index.to_series().diff().dropna()
    assert (steps == pd.Timedelta(days=1)).all(), "Ka dite qe mungojne ne kalendar"
    assert (daily["revenue"] >= 0).all(), "Ka te ardhura negative"
    assert not daily.isna().any().any(), "Ka vlera qe mungojne"
    assert (daily.loc[~daily["is_open"], "revenue"] == 0).all(), "Dite pa fature me te ardhura"
    print("Kontrollet e serise ditore: OK")