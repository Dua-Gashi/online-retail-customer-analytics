import pandas as pd
import numpy as np

FEATURES = [
    "lag_1", "lag_2", "lag_7", "lag_14", "avg_7", "avg_14", "day_of_week", 
    "month", "is_weekend", "year_end_break", "year_sin", "year_cos"
]


def build_daily_series(sales, returns=None, drop_last_day=False):
    """Seria ditore: nje rresht per cdo dite kalendarike.

    revenue     = te ardhurat bruto te shitjeve te dites
    refunds     = vlera e anulimeve te dites (negative ose zero)
    net_revenue = revenue + refunds
    orders      = numri i faturave unike te shitjeve
    is_open     = True nese ka pasur te pakten nje fature shitjeje
    Ditet pa shitje marrin 0 ne te gjitha kolonat numerike.
    """
    parts = [
        sales.groupby("Date")["Revenue"].sum().rename("revenue"),
        sales.groupby("Date")["Invoice"].nunique().rename("orders"),
    ]
    if returns is not None:
        parts.append(returns.groupby("Date")["Revenue"].sum().rename("refunds"))
    daily = pd.concat(parts, axis=1).sort_index()
    if "refunds" not in daily.columns:
        daily["refunds"] = 0.0

    if drop_last_day:
        daily = daily.iloc[:-1]

    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D", name="Date")
    daily = daily.reindex(full_index)

    daily["is_open"] = daily["orders"].notna()
    cols = ["revenue", "orders", "refunds"]
    daily[cols] = daily[cols].fillna(0)
    daily["orders"] = daily["orders"].astype(int)
    daily["net_revenue"] = daily["revenue"] + daily["refunds"]
    return daily


def validate_daily(daily):
    """Kontrollon qe seria ditore eshte e plote dhe e sakte."""
    assert daily.index.is_unique and daily.index.is_monotonic_increasing, "Indeksi nuk eshte unik ose i renditur"
    steps = daily.index.to_series().diff().dropna()
    assert (steps == pd.Timedelta(days=1)).all(), "Ka dite qe mungojne ne kalendar"
    assert (daily["revenue"] >= 0).all(), "Ka te ardhura bruto negative"
    assert (daily["refunds"] <= 0).all(), "Anulimet duhet te jene negative ose zero"
    assert not daily.isna().any().any(), "Ka vlera qe mungojne"
    assert (daily.loc[~daily["is_open"], "revenue"] == 0).all(), "Dite pa fature me te ardhura bruto"
    print("Kontrollet e serise ditore: OK")
    print("Dite me te ardhura neto negative:", int((daily["net_revenue"] < 0).sum()))
    

def features_for_dates(data, target_col="net_revenue"):
    """Krijon vecorite (X) dhe objektivin (target) nga seria ditore
    
    Cdo veqori per diten D perdor vetem informacion para dites D (lag, mesatare)
    ose informacion qe dihet paraprakisht nga kalendari (dita e javes, muaji, pushimi)
    """
    
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Indeksi duhet te jete DatetimeIndex")
    
    s = data[target_col].astype(float)
    ix = data.index
    
    f = pd.DataFrame(index=ix)
    f["lag_1"] = s.shift(1)
    f["lag_2"] = s.shift(2)
    f["lag_7"] = s.shift(7)
    f["lag_14"] = s.shift(14)
    f["avg_7"] = s.shift(1).rolling(7, min_periods=7).mean()
    f["avg_14"] = s.shift(1).rolling(14, min_periods=14).mean()
    f["day_of_week"] = ix.dayofweek
    f["month"] = ix.month
    f["is_weekend"] = (ix.dayofweek >= 5).astype(int)
    f["year_end_break"] = (((ix.month == 12) & (ix.day >= 24)) | ((ix.month == 1) & (ix.day <= 3))).astype(int)
    f["year_sin"] = np.sin(2 * np.pi * ix.dayofyear / 365.25)
    f["year_cos"] = np.cos(2 * np.pi * ix.dayofyear / 365.25)
    f["target"] = s
    return f

def make_dataset(daily):
    """Kthen tabelen e gatshme per model: vecorit + target, pa rreshta me vlera qe mungojne"""
    
    f = features_for_dates(daily)
    return f.dropna(subset=FEATURES + ["target"]) 

def validate_features(f):
    """Kontrollon qe vecorit perdorin vetem te dhena nga e kaluara (pa rrjedhje te dhenash)"""
    
    t = f["target"].to_numpy()
    assert np.allclose(f["lag_1"].to_numpy()[1:], t[:-1]), "lag_1 nuk eshte vlera e dites se meparshme"
    assert np.allclose(f["lag_7"].to_numpy()[7:], t[:-7]), "lag_7 nuk eshte vlera e 7 diteve me pare"
    for i in (20, 100, 300):
        assert np.isclose(f["avg_7"].iloc[i], t[i - 7:i].mean()), "avg_7 perdor diten aktuale ose te ardhmen"    
    assert f["day_of_week"].between(0, 6).all(), "dita e javes jashte intervalit"
    print("Kontrollet e vecorive: OK")    
        