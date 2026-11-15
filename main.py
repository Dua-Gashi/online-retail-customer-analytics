from src.data import(PROCESSED_DIR, load_data, clean_data, add_columns, validate_data, save_processed)
from src.features import build_rfm, validate_rfm

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
    
    
    print("\nPermbledhje\n")
    print(f"Periudha: {sales["Date"].min().date()} -> {sales["Date"].max().date()}")
    print(f"Fatura: {sales["Invoice"].nunique():,}")
    print(f"Klientet me ID: {sales["Customer ID"].nunique():,}")
    print(f"Te ardhurat gjithsej: {sales["Revenue"].sum():,.0f} £")
    
if __name__ == "__main__":
    main()