"""
Retail Sales Performance Analysis
Run: python analyze_sales.py
Creates a synthetic dataset, summary CSV files and charts in outputs/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def make_data(n=1200, seed=21):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    products = ["Laptop", "Phone", "Headphones", "Monitor", "Keyboard"]
    regions = ["North", "South", "East", "West"]
    df = pd.DataFrame({
        "order_id": np.arange(10001, 10001+n),
        "order_date": rng.choice(dates, n),
        "product": rng.choice(products, n, p=[.16,.28,.22,.14,.20]),
        "region": rng.choice(regions, n),
        "quantity": rng.integers(1, 6, n),
    })
    price = {"Laptop": 52000, "Phone": 24000, "Headphones": 2200,
             "Monitor": 11500, "Keyboard": 1300}
    df["unit_price"] = df["product"].map(price)
    df["unit_price"] = (df["unit_price"] * rng.uniform(.90, 1.10, n)).round(2)
    df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)
    return df

def main():
    df = make_data()
    df["order_date"] = pd.to_datetime(df["order_date"])
    df.to_csv(OUT / "sales_data.csv", index=False)

    monthly = (df.assign(month=df["order_date"].dt.to_period("M").astype(str))
                 .groupby("month", as_index=False)["revenue"].sum())
    product = (df.groupby("product", as_index=False)
                 .agg(revenue=("revenue", "sum"), units_sold=("quantity", "sum"))
                 .sort_values("revenue", ascending=False))
    region = (df.groupby("region", as_index=False)["revenue"].sum()
                .sort_values("revenue", ascending=False))

    monthly.to_csv(OUT / "monthly_revenue.csv", index=False)
    product.to_csv(OUT / "product_performance.csv", index=False)
    region.to_csv(OUT / "regional_performance.csv", index=False)

    print("Total revenue:", round(df["revenue"].sum(), 2))
    print("Number of orders:", len(df))
    print("\nTop products by revenue:\n", product.to_string(index=False))
    print("\nRevenue by region:\n", region.to_string(index=False))

    plt.figure(figsize=(9, 4))
    plt.plot(monthly["month"], monthly["revenue"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("Monthly Revenue")
    plt.xlabel("Month"); plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(OUT / "monthly_revenue.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(product["product"], product["revenue"])
    plt.title("Revenue by Product")
    plt.xlabel("Product"); plt.ylabel("Revenue")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "product_revenue.png", dpi=160)
    plt.close()

if __name__ == "__main__":
    main()
