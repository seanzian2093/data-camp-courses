import os
from utils import get_connection

conn = get_connection()


def create_professors_table(cursor):
    """Create the university_professors table if it doesn't exist"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS university_professors (
        firstname VARCHAR(100),
        lastname VARCHAR(100),
        university VARCHAR(200),
        university_shortname VARCHAR(200),
        university_city VARCHAR(200),
        function VARCHAR(200),
        organization VARCHAR(200),
        organization_sector VARCHAR(200)
    );
    """
    cursor.execute(create_table_query)


def import_professors_data_with_copy(conn, file_path):
    """Import data using PostgreSQL COPY statement"""
    try:
        cursor = conn.cursor()

        # Create table
        create_professors_table(cursor)

        # Clear existing data (optional)
        cursor.execute("DELETE FROM university_professors;")

        # Get absolute path for COPY command
        abs_file_path = os.path.abspath(file_path)

        # COPY command - adjust delimiter and options based on your file format
        copy_query = f"""
        COPY university_professors (firstname, lastname, university, university_shortname, university_city, function, organization, organization_sector)
        FROM '{abs_file_path}'
        WITH (FORMAT CSV, DELIMITER E'\\t', HEADER TRUE);
        """

        cursor.execute(copy_query)
        conn.commit()
        cursor.close()
        print(f"Data imported successfully using COPY from {file_path}")
        return True

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return False
    except Exception as e:
        print(f"Error importing data with COPY: {e}")
        conn.rollback()
        return False


def import_professors_data_with_copy_from(conn, file_path):
    """Alternative method using copy_from() which works with file objects"""
    try:
        cursor = conn.cursor()

        # Create table
        create_professors_table(cursor)

        # Clear existing data (optional)
        cursor.execute("DELETE FROM university_professors;")

        # Use copy_from with file object
        with open(file_path, "r", encoding="utf-8") as file:
            # Skip header if present
            next(file)

            cursor.copy_from(
                file,
                "university_professors",
                columns=(
                    "firstname",
                    "lastname",
                    "university",
                    "university_shortname",
                    "university_city",
                    "function",
                    "organization",
                    "organization_sector",
                ),
                sep="\t",  # Change to ',' for comma-separated files
            )

        conn.commit()
        cursor.close()
        print(f"Data imported successfully using copy_from() from {file_path}")
        return True

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return False
    except Exception as e:
        print(f"Error importing data with copy_from: {e}")
        conn.rollback()
        return False


def query_professors(conn):
    """Query and display some records from the table"""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM university_professors;")
        count = cursor.fetchone()[0]
        print(f"Total records in table: {count}")

        cursor.execute("SELECT * FROM university_professors LIMIT 5;")
        records = cursor.fetchall()
        print("\nFirst 5 records:")
        for record in records:
            print(record)

        cursor.close()
    except Exception as e:
        print(f"Error querying data: {e}")


# Main execution
conn = get_connection()
if conn:
    print("Connection to the PostgreSQL established successfully.")

    # Import data using COPY method
    file_path = "data/university_professors.txt"

    # Try the copy_from method first (more reliable)
    if import_professors_data_with_copy_from(conn, file_path):
        query_professors(conn)
    else:
        # Fallback to direct COPY command
        print("Trying direct COPY command...")
        if import_professors_data_with_copy(conn, file_path):
            query_professors(conn)

    conn.close()
else:
    print("Connection to the PostgreSQL encountered an error.")
