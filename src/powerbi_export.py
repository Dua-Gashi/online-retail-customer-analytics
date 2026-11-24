from pathlib import Path

import numpy as np
import pandas as pd

POWERBI_DIR = Path(__file__).resolve().parent.parent / "powerbi" / "data"
SEGMENT_ORDER = {"VIP": 1, "Champions": 2, "At risk": 3, "Recent": 4, "Lost": 5}

def daily_table(daily):
    """Seria ditore me kolona ndihmese per Power BI"""
    out = daily.reset_index()
    out["Year"] = out["Date"].dt.year
    out["YearMonth"] = out["Date"].dt.strftime("%Y-%m")
    out["Weekday"] = out["Date"].dt.day_name()
    out["WeekdayNum"] = out["Date"].dt.dayofweek + 1
    return out

def monthly_table(sales, returns, last_date):
    """Nje rresht per muaj, deri te data e fundit e plote"""
    
    s = sales[sales["Date"] <= last_date]
    r = returns[returns["Date"] <= last_date]
    monthly = pd.concat(
        [
            s.groupby("YearMonth")["Revenue"].sum().rename("revenue"),
            r.groupby("YearMonth")["Revenue"].sum().rename("refunds"),
            s.groupby("YearMonth")["Invoice"].nunique().rename("orders"),
            s.dropna(subset=["Customer ID"]).groupby("YearMonth")["Customer ID"].nunique().rename("active_customers"),
        ],
        axis=1,
    ).fillna(0).sort_index()
    monthly["net_revenue"] = monthly["revenue"] + monthly["refunds"]
    monthly = monthly.astype({"orders": int, "active_customers": int}).reset_index()
    monthly["MonthStart"] = pd.to_datetime(monthly["YearMonth"] + "-01")
    return monthly

def country_table(sales, last_date):
    """Te ardhurat, porosite dhe klientet per vend"""
    s = sales[sales["Date"] <= last_date]
    return (
        s.groupby("Country")
        .agg(revenue=("Revenue", "sum"), orders=("Invoice", "nunique"), customers=("Customer ID", "nunique"))
        .sort_values("revenue", ascending=False)
        .reset_index()
    )
    
def timeline_table(daily, forecast):
    """Te ardhurat reale dhe parashikimi ne nje tabele; parashikimi fillon nga pika e fundit reale"""
    
    actual = daily["net_revenue"].rename("actual")
    fc = forecast["forecast"].rename("forecast")
    timeline = pd.concat([actual, fc], axis=1).sort_index()
    timeline.index.name = "Date"
    last_actual = actual.index.max()
    timeline.loc[last_actual, "forecast"] = timeline.loc[last_actual, "actual"]
    return timeline.reset_index()

def export_powerbi_tables(sales, returns, segments, daily, forecast, folder=POWERBI_DIR):
    """Ruan tabelat CSV per dashborad-in ne Power BI"""
    
    folder.mkdir(parents=True, exist_ok=True)
    last_date = daily.index.max()
    
    tables = {
        "daily.csv": daily_table(daily),
        "monthly.csv": monthly_table(sales, returns, last_date),
        "countries.csv": country_table(sales, last_date),
        "timeline.csv": timeline_table(daily, forecast),
        "forecast.csv": forecast.reset_index().rename(columns={"date": "Date", "day": "Weekday"}),
    }
    for name, table in tables.items():
        table.to_csv(folder / name, index=False)
        
    seg = segments.reset_index()
    seg["SegmentOrder"] = seg["Segment"].map(SEGMENT_ORDER)
    seg.to_csv(folder / "customers_segments.csv", index=False)

    print(f"U ruajten {len(tables) + 1} tabela te {folder}")


def validate_export(folder=POWERBI_DIR):
    """Kontrollon qe tabelat e eksportuara jane te qendrueshme me njera-tjetren."""
    daily = pd.read_csv(folder / "daily.csv")
    monthly = pd.read_csv(folder / "monthly.csv")
    seg = pd.read_csv(folder / "customers_segments.csv")
    assert np.isclose(daily["net_revenue"].sum(), monthly["net_revenue"].sum()), "Te ardhurat neto: ditore dhe mujore nuk perputhen"
    assert daily["orders"].sum() == monthly["orders"].sum(), "Porosite: ditore dhe mujore nuk perputhen"
    assert seg["Customer ID"].is_unique, "Customer ID i perseritur"
    assert seg["Segment"].notna().all() and seg["SegmentOrder"].notna().all(), "Ka kliente pa segment"
    print("Kontrollet e eksportit: OK")