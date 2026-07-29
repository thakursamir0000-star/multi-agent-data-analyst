"""
Generate a synthetic sales dataset for demo and testing.

Run this script to create sample_data/sales_sample.csv.
"""

import csv
import random
import os
from datetime import datetime, timedelta

# Seed for reproducibility
random.seed(42)

# Configuration
NUM_ROWS = 500
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "sales_sample.csv")

PRODUCTS = [
    ("Widget A", "Widgets", 12.99),
    ("Widget B", "Widgets", 18.50),
    ("Widget Pro", "Widgets", 34.99),
    ("Gadget X", "Gadgets", 49.99),
    ("Gadget Y", "Gadgets", 79.99),
    ("Gadget Z", "Gadgets", 129.00),
    ("Service Basic", "Services", 99.00),
    ("Service Premium", "Services", 249.00),
    ("Accessory S", "Accessories", 5.99),
    ("Accessory M", "Accessories", 14.99),
    ("Accessory L", "Accessories", 29.99),
]

REGIONS = ["North", "South", "East", "West"]

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 6, 30)
DATE_RANGE_DAYS = (END_DATE - START_DATE).days


def generate_row(row_id: int) -> dict:
    product_name, category, unit_price = random.choice(PRODUCTS)
    region = random.choice(REGIONS)
    date = START_DATE + timedelta(days=random.randint(0, DATE_RANGE_DAYS))
    quantity = random.randint(1, 50)
    discount = round(random.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20, 0.25]), 2)
    revenue = round(quantity * unit_price * (1 - discount), 2)
    cost = round(revenue * random.uniform(0.4, 0.75), 2)
    profit = round(revenue - cost, 2)

    # Inject a few nulls for realism (~3% chance)
    if random.random() < 0.03:
        discount = ""
    if random.random() < 0.02:
        region = ""

    return {
        "date": date.strftime("%Y-%m-%d"),
        "product": product_name,
        "category": category,
        "region": region if region else "",
        "quantity": quantity,
        "unit_price": unit_price,
        "discount": discount,
        "revenue": revenue,
        "cost": cost,
        "profit": profit,
    }


def main():
    rows = [generate_row(i) for i in range(NUM_ROWS)]
    fieldnames = ["date", "product", "category", "region", "quantity",
                  "unit_price", "discount", "revenue", "cost", "profit"]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {NUM_ROWS} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
