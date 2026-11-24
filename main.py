from src.data import(
    PROCESSED_DIR, 
    load_data, 
    clean_data, 
    add_columns, 
    validate_data, 
    save_processed
    )
from src.features import build_rfm, validate_rfm
from src.segmentation import(
    final_segmentation,
    profile_cluster,
    validate_segments,
    save_segments
)
from src.forecasting import (
    build_daily_series,
    validate_daily,
    features_for_dates,
    validate_features,
    compare_models,
    train_final_model,
    predict_future,
    save_forecast_figure,
) 
from src.powerbi_export import export_powerbi_tables, validate_export

def main():
    print("1. Ngarkimi i te dhenave")
    raw = load_data()
    print(f"Rreshta: {len(raw):,} | Kolona: {raw.shape[1]}")
    
    print("\n2. Pastrimi i te dhenave")
    sales, returns = clean_data(raw)
    
    print("\n3. Kolonat e reja dhe kontrollet")
    sales = add_columns(sales)
    returns = add_columns(returns)
    validate_data(sales, returns)
    
    print("\n4. Ruajtja")
    save_processed(sales, returns)
    
    print("\n5. Tabela RFM")
    rfm = build_rfm(sales)
    validate_rfm(rfm)
    rfm.to_csv(PROCESSED_DIR / "rfm.csv")
    print(f"Klientet me RFM: {len(rfm):,}")
    print(rfm.describe().round(1))
    
    print("\n6. Segmentimi i klienteve")
    segments = final_segmentation(rfm)
    validate_segments(segments, rfm)
    profile = profile_cluster(rfm, segments["Segment"]).sort_values("Te_ardhurat", ascending=False)
    print(profile.to_string()) 
    save_segments(segments)  
    
    print("\n7. Parashikimi i shitjeve ditore")
    daily = build_daily_series(sales, returns, drop_last_day=True)
    validate_daily(daily)
    validate_features(features_for_dates(daily))

    comparison, _ = compare_models(daily)
    print(comparison.sort_values("MAE_dite_me_shitje").round(0).to_string())

    final_model = train_final_model(daily)
    forecast = predict_future(daily, final_model, days=7)
    print(forecast.to_string())
    save_forecast_figure(daily, forecast)
    
    print("\n8. Eksporti per Power BI")
    export_powerbi_tables(sales, returns, segments, daily, forecast)
    validate_export()
    
    print("\nPermbledhje\n")
    print(f"Periudha: {sales["Date"].min().date()} -> {sales["Date"].max().date()}")
    print(f"Fatura: {sales["Invoice"].nunique():,}")
    print(f"Klientet me ID: {sales["Customer ID"].nunique():,}")
    print(f"Te ardhurat gjithsej: {sales["Revenue"].sum():,.0f} £")
    
if __name__ == "__main__":
    main()