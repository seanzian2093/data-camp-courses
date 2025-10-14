import psycopg2


def get_connection():
    try:
        return psycopg2.connect(
            database="postgres",
            user="postgres",
            password="zdxzdxzdx",
            host="127.0.0.1",
            port=5433,
        )
    except:
        return False


if __name__ == "__main__":

    conn = get_connection()
    if conn:
        print(f"Connection to the PostgreSQL established successfully - {conn}")
    else:
        print("Failed to connect to the PostgreSQL database.")
