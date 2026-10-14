"""
Superstore Sales & Profit Analysis
----------------------------------
Step 1: Load data
Step 2: Clean data
Step 3: Create new columns (feature engineering)
Step 4: Explore data + save charts
Step 5: Save cleaned CSV and load it into MySQL

Install first:
    pip install pandas matplotlib seaborn sqlalchemy pymysql

Put the Kaggle file next to this script and keep its name as:
    Sample - Superstore.csv
(If yours has a different name, change CSV_FILE below.)
"""

import os
from pathlib import Path
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CSV_FILE = r"C:\Users\owner.DESKTOP-R31V7AU\OneDrive\VS_CODE\Sample - Superstore.csv"

sns.set_style("whitegrid")

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
os.makedirs("charts", exist_ok=True)

# ------------------------------------------------------------------
# STEP 1: LOAD DATA
# ------------------------------------------------------------------
# latin-1 encoding is needed because the Kaggle file has special characters
df = pd.read_csv(CSV_FILE, encoding="latin-1")

print("Shape (rows, columns):", df.shape)
print(df.head())
df.info()
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ------------------------------------------------------------------
# STEP 2: CLEAN DATA
# ------------------------------------------------------------------
# 2.1 Make column names SQL-friendly: "Order Date" -> "order_date"
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

# 2.2 Remove duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"\nDuplicates removed: {before - len(df)}")

# 2.3 Convert date columns from text to real dates
# If you get an error here, try: pd.to_datetime(df["order_date"], dayfirst=True)
df["order_date"] = pd.to_datetime(df["order_date"], format="mixed")
df["ship_date"] = pd.to_datetime(df["ship_date"], format="mixed")

# 2.4 Postal code is an ID, not a number to calculate with
df["postal_code"] = df["postal_code"].astype(str)

# 2.5 Remove extra spaces in text columns
for col in ["segment", "region", "category", "sub_category", "ship_mode"]:
    df[col] = df[col].str.strip()

# ------------------------------------------------------------------
# STEP 3: CREATE NEW COLUMNS
# ------------------------------------------------------------------
df["order_year"] = df["order_date"].dt.year
df["order_month"] = df["order_date"].dt.month
df["month_name"] = df["order_date"].dt.month_name()
df["delivery_days"] = (df["ship_date"] - df["order_date"]).dt.days
df["profit_margin"] = (df["profit"] / df["sales"] * 100).round(2)

# Check outliers in profit (we KEEP them: they are real big orders / big losses)
q1, q3 = df["profit"].quantile([0.25, 0.75])
iqr = q3 - q1
outliers = df[(df["profit"] < q1 - 1.5 * iqr) | (df["profit"] > q3 + 1.5 * iqr)]
print(f"\nProfit outliers (kept in data): {len(outliers)}")

# ------------------------------------------------------------------
# STEP 4: EXPLORE DATA (EDA) + SAVE CHARTS
# ------------------------------------------------------------------
# 4.1 Key numbers (KPIs)
total_sales = df["sales"].sum()
total_profit = df["profit"].sum()
total_orders = df["order_id"].nunique()
total_customers = df["customer_id"].nunique()
print("\n===== KEY NUMBERS =====")
print(f"Total Sales     : {total_sales:,.2f}")
print(f"Total Profit    : {total_profit:,.2f}")
print(f"Profit Margin   : {total_profit / total_sales * 100:.2f}%")
print(f"Total Orders    : {total_orders}")
print(f"Total Customers : {total_customers}")
print(f"Avg Delivery    : {df['delivery_days'].mean():.1f} days")

# 4.2 Monthly sales trend
monthly = df.groupby(df["order_date"].dt.to_period("M"))["sales"].sum()
monthly.index = monthly.index.to_timestamp()
plt.figure(figsize=(12, 5))
plt.plot(monthly.index, monthly.values, marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("charts/01_monthly_sales.png", dpi=150)
plt.close()

# 4.3 Sales and profit by category
cat = df.groupby("category")[["sales", "profit"]].sum()
cat.plot(kind="bar", figsize=(8, 5))
plt.title("Sales and Profit by Category")
plt.ylabel("Amount")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("charts/02_category_sales_profit.png", dpi=150)
plt.close()

# 4.4 Profit by sub-category (red = loss, green = profit)
sub = df.groupby("sub_category")["profit"].sum().sort_values()
colors = ["red" if v < 0 else "green" for v in sub.values]
plt.figure(figsize=(9, 7))
sub.plot(kind="barh", color=colors)
plt.title("Total Profit by Sub-Category")
plt.xlabel("Profit")
plt.tight_layout()
plt.savefig("charts/03_subcategory_profit.png", dpi=150)
plt.close()

# 4.5 Profit by region
region = df.groupby("region")["profit"].sum().sort_values(ascending=False)
plt.figure(figsize=(7, 5))
sns.barplot(x=region.index, y=region.values)
plt.title("Profit by Region")
plt.ylabel("Profit")
plt.tight_layout()
plt.savefig("charts/04_region_profit.png", dpi=150)
plt.close()

# 4.6 Does discount hurt profit?
plt.figure(figsize=(8, 5))
plt.scatter(df["discount"], df["profit"], alpha=0.3)
plt.title("Discount vs Profit")
plt.xlabel("Discount")
plt.ylabel("Profit")
plt.tight_layout()
plt.savefig("charts/05_discount_vs_profit.png", dpi=150)
plt.close()

bands = pd.cut(
    df["discount"],
    bins=[-0.01, 0, 0.2, 0.4, 1],
    labels=["0%", "1-20%", "21-40%", "40%+"],
)
band_profit = df.groupby(bands, observed=True)["profit"].mean()
print("\nAverage profit per order by discount band:\n", band_profit.round(2))

plt.figure(figsize=(7, 5))
sns.barplot(x=band_profit.index.astype(str), y=band_profit.values)
plt.title("Average Profit per Order by Discount Band")
plt.xlabel("Discount band")
plt.ylabel("Average profit")
plt.tight_layout()
plt.savefig("charts/06_discount_band_profit.png", dpi=150)
plt.close()

# 4.7 Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(
    df[["sales", "quantity", "discount", "profit"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("charts/07_correlation.png", dpi=150)
plt.close()

print("\nCharts saved in the 'charts' folder.")

# ------------------------------------------------------------------
# STEP 5: SAVE CLEANED DATA + LOAD INTO MYSQL
# ------------------------------------------------------------------
df.to_csv("superstore_cleaned.csv", index=False)
print("Cleaned file saved: superstore_cleaned.csv")

# First run this in MySQL:  CREATE DATABASE superstore_db;
# Never upload your real password to GitHub!
USER = "root"
PASSWORD = "your_password"
HOST = "localhost"
DATABASE = "superstore_db"

try:
    from sqlalchemy import create_engine

    engine = create_engine(
        f"mysql+pymysql://{USER}:{quote_plus(PASSWORD)}@{HOST}:3306/{DATABASE}"
    )
    df.to_sql("orders", engine, if_exists="replace", index=False, chunksize=1000)
    print("Data loaded into MySQL table: orders")
except Exception as e:
    print("MySQL load skipped/failed:", e)
    print("You can still import superstore_cleaned.csv using MySQL Workbench.")
