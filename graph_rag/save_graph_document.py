from create_node import book, jesus, jim
from create_relationship import written_by_jesus, written_by_jim
from helpers import NEO4J_PASSWORD, NEO4J_URL, NEO4J_USERNAME
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument

# Connect to the Neo4j Database
graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# Define the collection of graph documents
graph_document = GraphDocument(
    nodes=[book, jesus, jim], 
    relationships=[written_by_jesus, written_by_jim]
)

# Save the data to your graph
graph.add_graph_documents([graph_document])

if __name__ == "__main__":
    graph.refresh_schema()
    print(graph.schema)