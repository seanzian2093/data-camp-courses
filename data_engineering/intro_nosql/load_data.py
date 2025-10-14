import csv
import os
from utils import get_connection
from psycopg2.extras import execute_values


def create_disneyland_reviews_table(cursor):
    """Create the disneyland_reviews table if it doesn't exist"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS disneyland_reviews (
        review_id BIGINT,
        rating INTEGER,
        year_month VARCHAR(20),
        reviewer_location VARCHAR(200),
        review_text TEXT,
        branch VARCHAR(100)
    );
    """
    cursor.execute(create_table_query)
    print("Table 'disneyland_reviews' created successfully!")


def load_csv_to_table(cursor, csv_file_path):
    """Load data from CSV file to the disneyland_reviews table"""

    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: File {csv_file_path} not found!")
        return False

    cursor.execute("DELETE FROM disneyland_reviews;")

    try:
        data_to_insert = []

        # Extra delimiter in Review_Text so must use csv.DictReader
        with open(csv_file_path, "r", encoding="utf-8") as file:
            csv_reader = csv.DictReader(file)

            for row in csv_reader:
                try:
                    data_to_insert.append(
                        (
                            int(row["Review_ID"]),
                            int(row["Rating"]),
                            row["Year_Month"],
                            row["Reviewer_Location"],
                            row["Review_Text"],
                            row["Branch"],
                        )
                    )
                except ValueError:
                    print(f"Skipping invalid row: {row}")
                    continue

        # Batch insert using execute_values
        insert_query = """
            INSERT INTO disneyland_reviews 
            (review_id, rating, year_month, reviewer_location, review_text, branch)
            VALUES %s
        """

        execute_values(
            cursor, insert_query, data_to_insert, template=None, page_size=1000
        )

        print(
            f"Data imported successfully! {len(data_to_insert)} rows inserted from {csv_file_path}"
        )
        return True

    except Exception as e:
        print(f"Error loading data: {e}")
        return False


def get_table_stats(cursor):
    """Get basic statistics about the loaded data"""
    try:
        # Count total rows
        cursor.execute("SELECT COUNT(*) FROM disneyland_reviews;")
        total_rows = cursor.fetchone()[0]

        # Count by branch
        cursor.execute(
            """
            SELECT branch, COUNT(*) as count 
            FROM disneyland_reviews 
            GROUP BY branch 
            ORDER BY count DESC;
        """
        )
        branch_counts = cursor.fetchall()

        # Average rating
        cursor.execute(
            "SELECT AVG(rating) FROM disneyland_reviews WHERE rating IS NOT NULL;"
        )
        avg_rating = cursor.fetchone()[0]

        print(f"\n--- Table Statistics ---")
        print(f"Total reviews: {total_rows}")
        print(
            f"Average rating: {avg_rating:.2f}" if avg_rating else "Average rating: N/A"
        )
        print(f"\nReviews by branch:")
        for branch, count in branch_counts:
            print(f"  {branch}: {count}")

    except Exception as e:
        print(f"Error getting table statistics: {e}")


def main():
    """Main function to execute the data loading process"""

    # Get database connection
    conn = get_connection()
    if not conn:
        print("Failed to connect to PostgreSQL database!")
        return

    try:
        cursor = conn.cursor()

        # Create table
        create_disneyland_reviews_table(cursor)

        # Load CSV data
        csv_file_path = os.path.join("data", "disneyland_reviews_data.csv")
        success = load_csv_to_table(cursor, csv_file_path)

        if success:
            # Commit the transaction
            conn.commit()
            print("Data committed to database successfully!")

            # Show table statistics
            get_table_stats(cursor)
        else:
            # Rollback on failure
            conn.rollback()
            print("Transaction rolled back due to errors.")

    except Exception as e:
        print(f"Database error: {e}")
        conn.rollback()

    finally:
        # Close connections
        cursor.close()
        conn.close()
        print("Database connection closed.")


if __name__ == "__main__":
    main()
