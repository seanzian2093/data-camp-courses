from utils import get_connection

conn = get_connection()
cursor = conn.cursor()

# Check column types and constraints

query_col_types = """
SELECT 
    column_name, 
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_name = 'professors'
ORDER BY ordinal_position;
"""

# Change multiple column types with separate statements
query_change_multiple_columns = """
ALTER TABLE professors
ALTER COLUMN university_shortname 
TYPE CHAR(30);

ALTER TABLE professors
ALTER COLUMN firstname 
TYPE VARCHAR(16)
USING SUBSTRING(firstname FROM 1 FOR 16);

ALTER TABLE professors
ALTER COLUMN firstname SET NOT NULL;

ALTER TABLE professors
ALTER COLUMN lastname 
TYPE VARCHAR(64);

ALTER TABLE professors
ALTER COLUMN lastname SET NOT NULL;

ALTER TABLE universities
ADD CONSTRAINT unique_shortname UNIQUE (university_shortname);
"""

# More detailed column information query
query_detailed_col_types = """
SELECT 
    column_name, 
    data_type,
    character_maximum_length,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_name = 'professors'
ORDER BY ordinal_position;
"""

cursor.execute(query_col_types)
result_col_types = cursor.fetchall()
print(f"Column types and constraints for 'professors': {result_col_types}")

cursor.execute(query_change_multiple_columns)
conn.commit()
cursor.execute(query_col_types)
result_col_types = cursor.fetchall()
print(f"\nColumn types and constraints for 'professors': {result_col_types}")


cursor.close()
conn.close()
