import psycopg2
import csv
import os
from utils import get_connection


def load_reviews_to_postgres(conn):
    """Load reviews.csv data into PostgreSQL table using SQL only"""

    try:

        cur = conn.cursor()

        # Read CSV to determine columns and create table
        with open("data/reviews.csv", "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # Get column names

            print(f"Columns found: {headers}")

            # Create columns SQL with specific types
            create_table_sql = """
            DROP TABLE IF EXISTS reviews;
            CREATE TABLE reviews (
    reviewid INTEGER,
    title VARCHAR(255),
    url TEXT,
    score NUMERIC(3,1),
    best_new_music BOOLEAN,
    author VARCHAR(255),
    author_type VARCHAR(255),
    pub_date DATE,
    pub_weekday INTEGER,
    pub_day INTEGER,
    pub_month INTEGER,
    pub_year INTEGER
);
            """

            cur.execute(create_table_sql)

            print("Table 'reviews' created successfully")

            # Prepare insert statement
            placeholders = ", ".join(["%s"] * len(headers))
            insert_sql = (
                f"INSERT INTO reviews ({', '.join(headers)}) VALUES ({placeholders})"
            )

            # Insert data row by row
            row_count = 0
            for row in reader:
                cur.execute(insert_sql, row)
                row_count += 1

            print(f"Inserted {row_count} rows into reviews table")

        # Commit changes
        conn.commit()
        print("Data successfully loaded to PostgreSQL table 'reviews'")

    except FileNotFoundError:
        print("Error: data/reviews.csv file not found")
    except psycopg2.Error as e:
        print(f"PostgreSQL error: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"Error loading data: {str(e)}")
    finally:
        if "cur" in locals():
            cur.close()
        if "conn" in locals():
            conn.close()


load_reviews_to_postgres(get_connection())
