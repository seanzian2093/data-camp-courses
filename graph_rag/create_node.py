from helpers import *
from langchain_neo4j.graphs.graph_document import Node

# Create a Book node with a unique identifier
book = Node(
    type="Book",
    id="building-knowledge-graphs",
    properties={"title": "Building Knowledge Graphs"}
)

jim = Node(
    type="Person",
    id="webber-jim",
    properties={}
)

jesus = Node(id='barrasa-jesus', type='Person', properties={})

if __name__ == "__main__":
    print(book.id, book.type, book.properties['title'])
    visualize_node(book)
