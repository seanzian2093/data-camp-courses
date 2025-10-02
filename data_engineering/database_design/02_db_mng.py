from utils import get_connection

conn = get_connection()
cursor = conn.cursor()

# Create a role
query_create_role = """
-- does not support if not exists for roles
CREATE ROLE data_scientist;

CREATE ROLE Marta LOGIN;

CREATE ROLE admin WITH CREATEDB CREATEROLE;

-- Grant privileges on a table or view to a role
GRANT UPDATE, INSERT ON reviews TO data_scientist;

-- Alter a role
ALTER ROLE Marta WITH PASSWORD 'secure_password';

-- Add Marta to the data_scientist role/group
GRANT data_scientist TO Marta;

-- Remove Marta from the data_scientist role/group
REVOKE data_scientist FROM Marta;

-- Revoke before drop
REVOKE UPDATE, INSERT ON reviews FROM data_scientist;

-- Drop a role
DROP ROLE IF EXISTS data_scientist, Marta, admin;
"""
cursor.execute(query_create_role)
conn.commit()

# query_drop_role = """
# DROP ROLE IF EXISTS data_scientist;
# """
# cursor.execute(query_drop_role)
# conn.commit()

# For vertical partitioning, there is no specific syntax in PostgreSQL.
# You have to create a new table with particular columns and copy the data there.
# Afterward, you can drop the columns you want in the separate partition.
# If you need to access the full table, you can do so by using a JOIN clause.

# For horizontal partitioning, you can use table inheritance or declarative partitioning.
query_hpartition = """
-- Create the parent table
DROP TABLE IF EXISTS reviews_parent CASCADE;
CREATE TABLE reviews_parent (
    reviewid INT,
    score NUMERIC(3,1),
    pub_year INTEGER
) PARTITION BY LIST (pub_year);

-- Create partitions for specific years
CREATE TABLE reviews_2020 PARTITION OF reviews_parent FOR VALUES IN (2010);
CREATE TABLE reviews_2021 PARTITION OF reviews_parent FOR VALUES IN (2011);
CREATE TABLE reviews_2022 PARTITION OF reviews_parent FOR VALUES IN (2012);

-- Insert data into the parent table, which will be routed to the appropriate partition
INSERT INTO reviews_parent (reviewid, score, pub_year)
SELECT reviewid, score, pub_year FROM reviews
WHERE pub_year IN (2010, 2011, 2012);
"""

cursor.execute(query_hpartition)
conn.commit()

query_review_parent = """
-- View the partitions
SELECT * FROM reviews_parent LIMIT 5;
"""

cursor.execute(query_review_parent)
records = cursor.fetchall()
print(f"Records in reviews_parent: {records}")
