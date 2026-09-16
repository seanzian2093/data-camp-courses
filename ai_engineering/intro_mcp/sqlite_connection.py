import sqlite3

# Connect to currencies.db
conn = sqlite3.connect("currencies.db")
conn.row_factory = sqlite3.Row

# Execute the query
cursor = conn.execute("SELECT code, name FROM currencies WHERE code = ? LIMIT 1", ("USD",))
row = cursor.fetchone()
print(dict(row))

# Close the connection
conn.close()