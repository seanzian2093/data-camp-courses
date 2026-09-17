from helpers import NEO4J_PASSWORD, NEO4J_URL, NEO4J_USERNAME

from langchain_neo4j import Neo4jGraph

graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)


# Find and create relationship
cypher = """
MERGE (p:Person {name: $name})
MERGE (n:Nationality {nation: $nation})
MERGE (p)-[:HAS_NATIONALITY]->(n)
RETURN p.name AS name
"""

people = [
    {"name": "William Shakespeare", "nation": "England"},
    {"name": "Charles Dickens", "nation": "England"},
    {"name": "Victor Hugo", "nation": "France"},
]

print("Add people and their nationalities")
for person in people:
    result = graph.query(cypher, person)
    for record in result:
        print(record['name'])

# Query by the new relationship
cypher = """MATCH (p:Person {name: $name})-[:HAS_NATIONALITY]->(n:Nationality)
RETURN p.name AS name, n.nation AS nation"""

result = graph.query(cypher, {"name": "William Shakespeare"})

for record in result:
    print(f"{record['name']} has nationality {record['nation']}")

cypher = """
MATCH (p:Person)-[:HAS_NATIONALITY]->(n:Nationality {nation: $nation})
RETURN p.name AS name
"""

result = graph.query(cypher, {"nation": "England"})

print(f"Those who have nationality England are: ")
for record in result:
    print('\t', record['name'])