from helpers import NEO4J_PASSWORD, NEO4J_URL, NEO4J_USERNAME

from langchain_neo4j import Neo4jGraph

graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

cypher = """MATCH (b:Book {title: $title})-[:WRITTEN_BY]->(p:Person)
RETURN p.name AS author"""


# Execute the Cypher statement
# result = graph.query(cypher, {"title": "hamlet"})
result = graph.query(cypher, {"title": "Macbeth"})

# Extract the author
for row in result:
    print(row['author'])