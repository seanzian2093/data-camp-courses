"""
Pseudo Data Generator for Looker E-commerce Tables
Based on specifications in _looker__models.yml
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random
import string
import uuid

# Initialize Faker
fake = Faker()
Faker.seed(42)  # For reproducible results
np.random.seed(42)
random.seed(42)

# Configuration
NUM_DISTRIBUTION_CENTERS = 10
NUM_USERS = 1000
NUM_PRODUCTS = 500
NUM_INVENTORY_ITEMS = 2000
NUM_ORDERS = 800
NUM_ORDER_ITEMS = 1500
NUM_EVENTS = 5000


def generate_distribution_centers(num_records=NUM_DISTRIBUTION_CENTERS):
    """Generate pseudo data for stg_looker__distribution_centers"""
    data = []

    distribution_center_names = [
        "West Coast Distribution Center",
        "East Coast Distribution Center",
        "Midwest Distribution Hub",
        "Southern Regional Center",
        "Pacific Northwest Warehouse",
        "Texas Distribution Facility",
        "Florida Fulfillment Center",
        "Chicago Logistics Hub",
        "Denver Mountain Center",
        "Atlanta Southeast Hub",
    ]

    for i in range(num_records):
        data.append(
            {
                "id": i + 1,
                "name": (
                    distribution_center_names[i]
                    if i < len(distribution_center_names)
                    else fake.company() + " Distribution Center"
                ),
                "latitude": round(fake.latitude(), 6),
                "longitude": round(fake.longitude(), 6),
            }
        )

    return pd.DataFrame(data)


def generate_users(num_records=NUM_USERS):
    """Generate pseudo data for stg_looker__users"""
    data = []
    traffic_sources = [
        "Search",
        "Email",
        "Facebook",
        "Organic",
        "Display",
    ]

    for i in range(num_records):
        profile = fake.profile()
        created_at = fake.date_time_between(start_date="-2y", end_date="now")

        data.append(
            {
                "id": i + 1,
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "email": fake.email(),
                "age": random.randint(18, 80),
                "gender": random.choice(["M", "F", None]),
                "state": fake.state_abbr(),
                "street_address": fake.street_address(),
                "postal_code": fake.postcode(),
                "city": fake.city(),
                "country": "USA",  # Assuming US-based for consistency
                "latitude": round(fake.latitude(), 6),
                "longitude": round(fake.longitude(), 6),
                "traffic_source": random.choice(traffic_sources),
                "created_at": created_at,
            }
        )

    return pd.DataFrame(data)


def generate_products(num_records=NUM_PRODUCTS):
    """Generate pseudo data for stg_looker__products (filling in the blank table)"""
    data = []
    categories = [
        "Electronics",
        "Clothing",
        "Home & Garden",
        "Sports",
        "Books",
        "Beauty",
        "Toys",
        "Automotive",
    ]
    departments = [
        "Electronics",
        "Fashion",
        "Home",
        "Sports & Outdoors",
        "Books & Media",
        "Beauty & Personal Care",
        "Toys & Games",
        "Automotive",
    ]
    brands = [
        "Apple",
        "Samsung",
        "Nike",
        "Adidas",
        "Sony",
        "LG",
        "Canon",
        "Dell",
        "HP",
        "Microsoft",
        "Amazon",
        "Google",
    ]

    distribution_center_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for i in range(num_records):
        category = random.choice(categories)
        department = random.choice(departments)
        brand = random.choice(brands)
        cost = round(random.uniform(5.0, 500.0), 2)
        retail_price = round(
            cost * random.uniform(1.5, 3.0), 2
        )  # Markup between 50% to 200%
        distribution_center_id = random.choice(distribution_center_ids)

        data.append(
            {
                "id": i + 1,
                "cost": cost,
                "category": category,
                "name": fake.catch_phrase() + f" {category}",
                "brand": brand,
                "retail_price": retail_price,
                "department": department,
                "sku": "".join(
                    random.choices(string.ascii_uppercase + string.digits, k=8)
                ),
                "distribution_center_id": distribution_center_id,
            }
        )

    return pd.DataFrame(data)


def generate_inventory_items(
    num_records=NUM_INVENTORY_ITEMS, products_df=None, distribution_centers_df=None
):
    """Generate pseudo data for stg_looker__inventory_items"""
    if products_df is None or distribution_centers_df is None:
        raise ValueError("Products and distribution centers DataFrames are required")

    data = []
    product_ids = products_df["id"].tolist()
    distribution_center_ids = distribution_centers_df["id"].tolist()

    for i in range(num_records):
        product_id = random.choice(product_ids)
        product_info = products_df[products_df["id"] == product_id].iloc[0]

        created_at = fake.date_time_between(start_date="-1y", end_date="now")
        # 70% chance the item is sold
        sold_at = (
            fake.date_time_between(start_date=created_at, end_date="now")
            if random.random() < 0.7
            else None
        )

        data.append(
            {
                "id": i + 1,
                "product_id": product_id,
                "created_at": created_at,
                "sold_at": sold_at,
                "cost": product_info["cost"],
                "product_category": product_info["category"],
                "product_name": product_info["name"],
                "product_brand": product_info["brand"],
                "product_retail_price": product_info["retail_price"],
                "product_department": product_info["department"],
                "product_sku": product_info["sku"],
                "product_distribution_center_id": random.choice(
                    distribution_center_ids
                ),
            }
        )

    return pd.DataFrame(data)


def generate_orders(num_records=NUM_ORDERS, users_df=None):
    """Generate pseudo data for stg_looker__orders"""
    if users_df is None:
        raise ValueError("Users DataFrame is required")

    data = []
    user_ids = users_df["id"].tolist()
    statuses = ["Complete", "Pending", "Cancelled", "Processing", "Shipped", "Returned"]

    for i in range(num_records):
        user_id = random.choice(user_ids)
        user_info = users_df[users_df["id"] == user_id].iloc[0]
        status = random.choice(statuses)

        created_at = fake.date_time_between(start_date="-6m", end_date="now")

        # Generate timestamps based on status
        shipped_at = None
        delivered_at = None
        returned_at = None

        if status in ["Complete", "Shipped", "Returned"]:
            shipped_at = created_at + timedelta(days=random.randint(1, 3))
            if status in ["Complete", "Returned"]:
                delivered_at = shipped_at + timedelta(days=random.randint(1, 7))
                if status == "Returned":
                    returned_at = delivered_at + timedelta(days=random.randint(1, 30))

        data.append(
            {
                "order_id": i + 1,
                "user_id": user_id,
                "status": status,
                "gender": user_info["gender"],
                "created_at": created_at,
                "returned_at": returned_at,
                "shipped_at": shipped_at,
                "delivered_at": delivered_at,
                "num_of_item": random.randint(1, 5),
            }
        )

    return pd.DataFrame(data)


def generate_order_items(
    num_records=NUM_ORDER_ITEMS,
    orders_df=None,
    users_df=None,
    products_df=None,
    inventory_items_df=None,
):
    """Generate pseudo data for stg_looker__order_items"""
    if any(df is None for df in [orders_df, users_df, products_df, inventory_items_df]):
        raise ValueError("All referenced DataFrames are required")

    data = []
    order_ids = orders_df["order_id"].tolist()

    for i in range(num_records):
        order_id = random.choice(order_ids)
        order_info = orders_df[orders_df["order_id"] == order_id].iloc[0]

        # Get available inventory items that are sold
        available_inventory = inventory_items_df[inventory_items_df["sold_at"].notna()]
        if len(available_inventory) == 0:
            continue

        inventory_item = available_inventory.sample(1).iloc[0]

        # Calculate sale price (usually between cost and retail price)
        cost = inventory_item["cost"]
        retail_price = inventory_item["product_retail_price"]
        sale_price = round(random.uniform(cost, retail_price), 2)

        data.append(
            {
                "id": i + 1,
                "order_id": order_id,
                "user_id": order_info["user_id"],
                "product_id": inventory_item["product_id"],
                "inventory_item_id": inventory_item["id"],
                "status": order_info["status"],
                "created_at": order_info["created_at"],
                "shipped_at": order_info["shipped_at"],
                "delivered_at": order_info["delivered_at"],
                "returned_at": order_info["returned_at"],
                "sale_price": sale_price,
            }
        )

    return pd.DataFrame(data)


def generate_events(num_records=NUM_EVENTS, users_df=None):
    """Generate pseudo data for stg_looker__events"""
    if users_df is None:
        raise ValueError("Users DataFrame is required")

    data = []
    user_ids = users_df["id"].tolist()
    event_types = [
        "home",
        "product",
        "cart",
        "department",
        "purchase",
        "cancel",
        "login",
        "logout",
    ]
    browsers = ["Chrome", "Firefox", "Safari", "Edge", "Opera"]
    traffic_sources = [
        "Search",
        "Email",
        "Adwords",
        "YouTube",
        "Facebook",
        "Organic",
        "Direct",
    ]

    # Generate session data
    sessions = {}
    for _ in range(num_records // 5):  # Average 5 events per session
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "user_id": random.choice(user_ids),
            "ip_address": fake.ipv4(),
            "browser": random.choice(browsers),
            "traffic_source": random.choice(traffic_sources),
        }

    session_ids = list(sessions.keys())

    for i in range(num_records):
        session_id = random.choice(session_ids)
        session_info = sessions[session_id]
        user_id = session_info["user_id"]
        user_info = users_df[users_df["id"] == user_id].iloc[0]

        data.append(
            {
                "id": i + 1,
                "user_id": user_id,
                "sequence_number": random.randint(1, 20),
                "session_id": session_id,
                "created_at": fake.date_time_between(start_date="-3m", end_date="now"),
                "ip_address": session_info["ip_address"],
                "city": user_info["city"],
                "state": user_info["state"],
                "postal_code": user_info["postal_code"],
                "browser": session_info["browser"],
                "traffic_source": session_info["traffic_source"],
                "uri": f"/{random.choice(event_types)}"
                + (f"/{random.randint(1, 100)}" if random.random() > 0.5 else ""),
                "event_type": random.choice(event_types),
            }
        )

    return pd.DataFrame(data)


def main():
    """Generate all pseudo data and save to CSV files"""
    print("Generating pseudo data for Looker e-commerce tables...")

    # Generate data in dependency order
    print("1. Generating distribution centers...")
    distribution_centers_df = generate_distribution_centers()

    print("2. Generating users...")
    users_df = generate_users()

    print("3. Generating products...")
    products_df = generate_products()

    print("4. Generating inventory items...")
    inventory_items_df = generate_inventory_items(
        products_df=products_df, distribution_centers_df=distribution_centers_df
    )

    print("5. Generating orders...")
    orders_df = generate_orders(users_df=users_df)

    print("6. Generating order items...")
    order_items_df = generate_order_items(
        orders_df=orders_df,
        users_df=users_df,
        products_df=products_df,
        inventory_items_df=inventory_items_df,
    )

    print("7. Generating events...")
    events_df = generate_events(users_df=users_df)

    # Save to CSV files
    print("\nSaving data to CSV files...")
    distribution_centers_df.to_csv("pseudo_distribution_centers.csv", index=False)
    users_df.to_csv("pseudo_users.csv", index=False)
    products_df.to_csv("pseudo_products.csv", index=False)
    inventory_items_df.to_csv("pseudo_inventory_items.csv", index=False)
    orders_df.to_csv("pseudo_orders.csv", index=False)
    order_items_df.to_csv("pseudo_order_items.csv", index=False)
    events_df.to_csv("pseudo_events.csv", index=False)

    # Print summary statistics
    print("\n" + "=" * 50)
    print("DATA GENERATION SUMMARY")
    print("=" * 50)
    print(f"Distribution Centers: {len(distribution_centers_df):,} records")
    print(f"Users: {len(users_df):,} records")
    print(f"Products: {len(products_df):,} records")
    print(f"Inventory Items: {len(inventory_items_df):,} records")
    print(f"Orders: {len(orders_df):,} records")
    print(f"Order Items: {len(order_items_df):,} records")
    print(f"Events: {len(events_df):,} records")
    print("\nFiles saved:")
    print("- pseudo_distribution_centers.csv")
    print("- pseudo_users.csv")
    print("- pseudo_products.csv")
    print("- pseudo_inventory_items.csv")
    print("- pseudo_orders.csv")
    print("- pseudo_order_items.csv")
    print("- pseudo_events.csv")

    # Show sample data
    print(f"\n" + "=" * 50)
    print("SAMPLE DATA PREVIEW")
    print("=" * 50)

    print("\nDistribution Centers (first 3 rows):")
    print(distribution_centers_df.head(3).to_string(index=False))

    print("\nUsers (first 3 rows):")
    print(
        users_df[["id", "first_name", "last_name", "email", "age", "gender", "state"]]
        .head(3)
        .to_string(index=False)
    )

    print("\nProducts (first 3 rows):")
    print(
        products_df[["id", "name", "category", "brand", "cost", "retail_price"]]
        .head(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
