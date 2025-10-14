import pandas as pd
from utils import get_connection

conn = get_connection()
cursor = conn.cursor()

# The -> operator will return the queried field as JSON, while ->> will return the queried field as text.
query0 = """
    SELECT 
        review -> 'location' AS location, 
        review ->> 'statement' AS statement 
    FROM disneyland_reviews;
"""

# Find the data type of the location field
query1 = """
SELECT
    json_typeof(review -> 'location') AS location_type
FROM disneyland_reviews;
"""

# Update the query to select the nested reviewer field
query2 = """
SELECT 
	review -> 'location' ->> 'branch' AS branch,
    review -> 'location' ->> 'reviewer' AS reviewer
FROM disneyland_reviews;
"""

# Build the query to select the rid and rating fields
query3 = """
SELECT
	review -> 'statement' AS customer_review 
FROM disneyland_reviews 
WHERE review -> 'location' ->> 'branch' = 'Disneyland_California';
"""

# Use the #> and #>> operators to query along nested fields
query4 = """
	SELECT 
    	json_typeof(review #> '{statement}'),
        review #>> '{location, branch}' AS branch,
        review #>> '{location, zipcode}' AS zipcode
    FROM disneyland_reviews;
"""

# Extract specific fields from the JSON column
query5 = """
    SELECT 
        json_extract_path(review, 'statement'),
        json_extract_path_text(review, 'location', 'reviewer')
    FROM disneyland_reviews
    WHERE json_extract_path_text(review, 'location', 'branch') = 'Disneyland_California';
"""


# data = pd.read_sql(query0, conn)
# data = pd.read_sql(query1, conn)
# data = pd.read_sql(query2, conn)
# data = pd.read_sql(query3, conn)
# data = pd.read_sql(query4, conn)
data = pd.read_sql(query5, conn)

print(data)
