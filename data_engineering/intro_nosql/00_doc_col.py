import pandas as pd
from utils import get_connection

conn = get_connection()
cursor = conn.cursor()

# Build a query to create a JSON-object
query0 = """
SELECT
	row_to_json(row(review_id, rating, year_month))
FROM disneyland_reviews;
"""

# results = pd.read_sql(query0, conn)
# print(results.head(10))

query1 = """
SELECT
    json_build_object(
        'year_month', year_month,
        'location', reviewer_location,
        'statement', review_text,
        'branch', branch
    ) AS review
FROM disneyland_reviews;
"""

# results = pd.read_sql(query1, conn)
# print(results.head(10))

query2 = """
ALTER TABLE disneyland_reviews
DROP COLUMN IF EXISTS review;

ALTER TABLE disneyland_reviews
ADD COLUMN review JSON;

UPDATE disneyland_reviews
SET review = json_build_object(
    'statement', review_text,
    'year_month', year_month,
    'location', json_build_object(
        'reviewer', reviewer_location,
        'branch', branch
    )
);
"""

cursor.execute(query2)
conn.commit()
print("Column 'review' added and populated successfully!")

# Verify the update
query3 = """
SELECT review
FROM disneyland_reviews
LIMIT 10;
"""
results = pd.read_sql(query3, conn)
print(results.head(10))

# Close the cursor and connection
cursor.close()
conn.close()
