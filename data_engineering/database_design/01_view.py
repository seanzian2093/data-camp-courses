from utils import get_connection

conn = get_connection()
cursor = conn.cursor()


def get_non_system_views():
    query_get_views = """
    SELECT * FROM information_schema.views
    WHERE table_schema NOT IN ('information_schema', 'pg_catalog');
    """
    cursor.execute(query_get_views)
    records = cursor.fetchall()
    print(f"Existing views: {records}")
    return records


# get_non_system_views()

# Create a view for reviews with a score above 9
query_create_view = """
CREATE OR REPLACE VIEW high_scores AS
SELECT * FROM reviews
WHERE score > 9;
"""
cursor.execute(query_create_view)
conn.commit()

# Query a view
query_count_high_scores = """
SELECT COUNT(*) FROM high_scores
"""
cursor.execute(query_count_high_scores)
count = cursor.fetchone()[0]
print(f"Number of high score reviews: {count}")

# get_non_system_views()

# materialized view
query_create_materialized_view = """
CREATE MATERIALIZED VIEW high_scores_mat AS
SELECT reviewid, score FROM reviews
WHERE score > 9;
"""
cursor.execute(query_create_materialized_view)
conn.commit()

cursor.execute("SELECT COUNT(*) FROM high_scores_mat;")
count_mat = cursor.fetchone()[0]
print(f"Number of high score reviews in materialized view: {count_mat}")

# Update and refresh materialized view
cursor.execute("INSERT INTO reviews (reviewid, score) VALUES (9999, 10);")
conn.commit()
cursor.execute("REFRESH MATERIALIZED VIEW high_scores_mat;")
cursor.execute("SELECT COUNT(*) FROM high_scores_mat;")
count_mat = cursor.fetchone()[0]
print(f"Number of high score reviews in materialized view: {count_mat}")


# Materialized views are not listed as views
get_non_system_views()
