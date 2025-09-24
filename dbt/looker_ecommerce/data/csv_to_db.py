import duckdb
import os


def load_csv_to_duckdb(schema="looker_ecommerce"):
    """Load CSV files to DuckDB database"""

    # Connect to the database file
    conn = duckdb.connect("../dev.duckdb")

    print("Connected to DuckDB database")

    # Create schema if it doesn't exist
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    print(f"✓ Created/verified schema: {schema}")

    csv_files = {
        "users": "pseudo_users.csv",
        "products": "pseudo_products.csv",
        "inventory_items": "pseudo_inventory_items.csv",
        "orders": "pseudo_orders.csv",
        "order_items": "pseudo_order_items.csv",
        "events": "pseudo_events.csv",
    }

    for table_name, csv_file in csv_files.items():
        if os.path.exists(csv_file):
            conn.execute(
                f"""
                CREATE OR REPLACE TABLE {schema}.{table_name} AS 
                SELECT * FROM read_csv_auto('{csv_file}')
            """
            )
            print(f"✓ Loaded {schema}.{table_name} table from {csv_file}")
        else:
            print(f"⚠ Warning: {csv_file} not found, skipping {schema}.{table_name}")

    conn.close()


def query_examples(schema="looker_ecommerce"):
    """Examples of querying the loaded data"""

    conn = duckdb.connect("../dev.duckdb")

    print("\n=== Query Examples ===")

    # Correct way to set default schema in DuckDB
    conn.execute(f"SET schema = '{schema}'")
    # Example queries
    queries = [
        (
            "Total users by state",
            """
            SELECT state, COUNT(*) as user_count 
            FROM users 
            GROUP BY state 
            ORDER BY user_count DESC 
            LIMIT 5
        """,
        ),
        (
            "Average product price by category",
            """
            SELECT category, 
                   AVG(retail_price) as avg_price,
                   COUNT(*) as product_count
            FROM products 
            GROUP BY category 
            ORDER BY avg_price DESC
        """,
        ),
        (
            "Order status distribution",
            """
            SELECT status, COUNT(*) as order_count
            FROM orders 
            GROUP BY status 
            ORDER BY order_count DESC
        """,
        ),
    ]

    for description, query in queries:
        try:
            print(f"\n{description}:")
            result = conn.execute(query).fetchall()
            for row in result:
                print(f"  {row}")
        except Exception as e:
            print(f"  Error: {e}")

    conn.close()


def check_schema_table():
    conn = duckdb.connect("../dev.duckdb")
    # Check current schema
    result = conn.execute("SELECT current_schema()").fetchone()
    print(f"Current schema: {result[0]}")  # Output: main

    # List all schemas
    # schemas = conn.execute("SHOW SCHEMAS").fetchall()
    # print("Available schemas:", schemas)  # Output: [('information_schema',), ('main',)]

    # List tables in main schema
    tables = conn.execute("SHOW TABLES").fetchall()
    print("Tables in main schema:", tables)

    conn.close()


def drop_tables(schema="main"):
    conn = duckdb.connect("../dev.duckdb")
    tables = ["users", "products", "inventory_items", "orders", "order_items", "events"]
    for table in tables:
        conn.execute(f"DROP TABLE IF EXISTS {schema}.{table}")
        print(f"Dropped table {schema}.{table}")
    conn.close()
    print("\n✓ All specified tables dropped")


if __name__ == "__main__":
    # Load CSV files to DuckDB
    load_csv_to_duckdb()

    # Run some example queries
    query_examples()

    # Check schema and tables
    # check_schema_table()

    # Drop tables if needed
    # drop_tables()
