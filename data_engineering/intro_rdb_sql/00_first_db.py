from utils import get_connection

conn = get_connection()

# Query the right table in information_schema
query_info = """
SELECT table_name, table_schema 
FROM information_schema.tables
WHERE table_schema = 'public'
"""

# Create a new table of professors
query_create_professors = """
DROP TABLE IF EXISTS professors;
CREATE TABLE professors (
    firstname text,
    lastname text
);
"""

# Add new colun to professors
query_add_column = """
ALTER TABLE professors
ADD COLUMN university_shortname text;
"""

# Migrate data to professors
query_insert_professors = """
INSERT INTO professors
SELECT DISTINCT firstname, lastname, university_shortname
FROM university_professors;
"""


# Create a new table of universities
query_create_universities = """
DROP TABLE IF EXISTS universities;
CREATE TABLE universities (
    university_shortname text,
    university text,
    university_city text
    );

INSERT INTO universities
SELECT DISTINCT university_shortname, university, university_city
FROM university_professors;
"""


# Create a new talbe of affiliations
query_create_affiliations = """
DROP TABLE IF EXISTS affiliations;
CREATE TABLE affiliations (
    firstname text,
    lastname text,
    university_shortname text,
    function text,
    organisation text)
"""

# Rename and delete columns
query_rename_delete_columns = """
ALTER TABLE affiliations
RENAME COLUMN organisation TO organization;
ALTER TABLE affiliations
DROP COLUMN university_shortname;
"""

# Insert data into affiliations
query_insert_affiliations = """
INSERT INTO affiliations
SELECT DISTINCT firstname, lastname, function, organization
FROM university_professors;
"""

# Create a new table of organizations
query_create_organizations = """
DROP TABLE IF EXISTS organizations;
CREATE TABLE organizations (
    organization text,
    organization_sector text
    );

INSERT INTO organizations
SELECT DISTINCT organization, organization_sector
FROM university_professors;
"""


cursor = conn.cursor()

cursor.execute(query_create_professors)
cursor.execute(query_add_column)
cursor.execute(query_insert_professors)
conn.commit()
cursor.execute("SELECT count(*) FROM professors")
result_professors = cursor.fetchall()
print(f"Professors in the database: {result_professors}")

cursor.execute(query_create_universities)
conn.commit()
cursor.execute("SELECT count(*) FROM universities")
result_universities = cursor.fetchall()
print(f"Universities in the database: {result_universities}")

cursor.execute(query_create_affiliations)
cursor.execute(query_rename_delete_columns)
cursor.execute(query_insert_affiliations)
conn.commit()
cursor.execute("SELECT count(*) FROM affiliations")
result_affiliations = cursor.fetchall()
print(f"Affiliations in the database: {result_affiliations}")

cursor.execute(query_create_organizations)
conn.commit()
cursor.execute("SELECT count(*) FROM organizations")
result_organizations = cursor.fetchall()
print(f"Organizations in the database: {result_organizations}")

cursor.execute(query_info)
result_info = cursor.fetchall()
print(f"Tables in the database: {result_info}")

cursor.close()
conn.close()
