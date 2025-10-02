from utils import get_connection

conn = get_connection()
cursor = conn.cursor()


# Add primary key to organizations table
query_add_pk_organizations = """
ALTER TABLE organizations
RENAME COLUMN organization TO id;

ALTER TABLE organizations
ADD CONSTRAINT organization_pk PRIMARY KEY (id);
"""

cursor.execute(query_add_pk_organizations)
conn.commit()

# Check for primary key columns
query_primary_keys = """
SELECT 
    kcu.column_name,
    tc.constraint_type
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu 
    ON tc.constraint_name = kcu.constraint_name 
    AND tc.table_schema = kcu.table_schema
WHERE tc.table_name = 'organizations' 
    AND tc.constraint_type = 'PRIMARY KEY';
"""

cursor.execute(query_primary_keys)
result_primary_keys_organizations = cursor.fetchall()
print(f"Primary key columns for 'organizations': {result_primary_keys_organizations}")

# Add a new id column to professors
query_add_pk_professors = """
ALTER TABLE professors
ADD COLUMN id SERIAL;

ALTER TABLE professors
ADD CONSTRAINT professors_pk PRIMARY KEY (id);
"""
cursor.execute(query_add_pk_professors)
conn.commit()

cursor.execute(query_primary_keys.replace("organizations", "professors"))
result_primary_keys_professors = cursor.fetchall()
print(f"Primary key columns for 'professors': {result_primary_keys_professors}")

# Add primary key constraint to universities table
query_add_pk_universities = """
ALTER TABLE universities
RENAME COLUMN university_shortname TO id;

ALTER TABLE universities
ADD CONSTRAINT universities_pk PRIMARY KEY (id);
"""

cursor.execute(query_add_pk_universities)
conn.commit()

# Add foreign key constraint to professors table
query_add_fk_professors = """
ALTER TABLE professors
RENAME COLUMN university_shortname TO university_id;

ALTER TABLE professors
ADD CONSTRAINT professors_fk
FOREIGN KEY (university_id)
REFERENCES universities(id);
"""

cursor.execute(query_add_fk_professors)
conn.commit()

cursor.execute(
    query_primary_keys.replace("PRIMARY", "FOREIGN").replace(
        "organizations", "professors"
    )
)
result_fk_professors = cursor.fetchall()
print(f"Foreign key columns for 'professors': {result_fk_professors}")
