from utils import get_connection

conn = get_connection()
cursor = conn.cursor()


# Try to insert a new record of professor
# We will get an error -
# psycopg2.errors.ForeignKeyViolation: insert or update on table "professors" violates foreign key constraint "professors_fk"
# DETAIL:  Key (university_id)=(MIT) is not present in table "universities".
query_insert_professor = """
INSERT INTO professors (firstname, lastname, university_id)
VALUES ('Albert', 'Einstein', 'MIT');
"""
# cursor.execute(query_insert_professor)
# conn.commit()

# cursor.execute("SELECT DISTINCT id FROM universities;")
cursor.execute("SELECT * FROM universities;")
records = cursor.fetchall()
print(f"Existing university IDs: {records}")

query_insert_professor = """
INSERT INTO professors (firstname, lastname, university_id)
VALUES ('Albert', 'Einstein', 'EPF');
"""
cursor.execute(query_insert_professor)
conn.commit()
