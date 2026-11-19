import pandas as pd


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