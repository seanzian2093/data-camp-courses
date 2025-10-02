import duckdb
from pathlib import Path


def read_txt_to_duckdb(
    txt_file_path, table_name, db_path=None, delimiter="\t", header=True
):
    """
    Read a txt file from data folder into a DuckDB table

    Args:
        txt_file_path (str): Path to the txt file relative to data folder
        table_name (str): Name of the table to create in DuckDB
        db_path (str): Path to DuckDB database file (optional, uses in-memory if None)
        delimiter (str): Delimiter used in the txt file (default: ',')
        header (bool): Whether the first row contains column names (default: True)

    Returns:
        duckdb.DuckDBPyConnection: Database connection object
    """
    # Get the data folder path
    current_dir = Path(__file__).parent
    data_folder = current_dir / "data"
    full_txt_path = data_folder / txt_file_path

    # Check if file exists
    if not full_txt_path.exists():
        raise FileNotFoundError(f"File not found: {full_txt_path}")

    # Create DuckDB connection
    if db_path:
        conn = duckdb.connect(db_path)
    else:
        conn = duckdb.connect()  # In-memory database

    try:
        # Read the txt file using DuckDB's CSV reader
        # DuckDB can handle various text formats through its CSV reader
        header_param = "TRUE" if header else "FALSE"

        query = f"""
        CREATE TABLE {table_name} AS 
        SELECT * FROM read_csv_auto('{full_txt_path}', 
                                   delim='{delimiter}', 
                                   header={header_param})
        """

        conn.execute(query)
        print(f"Successfully loaded {txt_file_path} into table '{table_name}'")

        # Show table info
        result = conn.execute(
            f"SELECT COUNT(*) as row_count FROM {table_name}"
        ).fetchone()
        print(f"Table '{table_name}' contains {result[0]} rows")

        return conn

    except Exception as e:
        print(f"Error loading file: {e}")
        conn.close()
        raise


def list_data_files():
    """List all files in the data folder"""
    current_dir = Path(__file__).parent
    data_folder = current_dir / "data"

    if not data_folder.exists():
        print("Data folder not found")
        return []

    files = [f.name for f in data_folder.iterdir() if f.is_file()]
    print("Files in data folder:")
    for file in files:
        print(f"  - {file}")
    return files


# Example usage
if __name__ == "__main__":
    # List available files
    list_data_files()

    # Example: Load a txt file (adjust the filename as needed)
    conn = read_txt_to_duckdb(
        "university_professors.txt", "university_professors", "dev.duckdb"
    )

    # Query the table
    result = conn.execute("SELECT * FROM university_professors LIMIT 5").fetchdf()
    print(result)

    # Don't forget to close the connection
    conn.close()
