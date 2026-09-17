from helpers import visualize_graph
from create_node import book, jesus, jim
from langchain_neo4j.graphs.graph_document import Node, Relationship

relationships = []

for author in [jesus, jim]:
  	# Create a Relationship between book and each author
    relationships.append(Relationship(
    	source=book, 
      	target=author, 
      	type="WRITTEN_BY"
    ))

written_by_jesus = Relationship(
    # source=Node(id='building-knowledge-graphs', type='Book', properties={}),
    # target=Node(id='barrasa-jesus', type='Person', properties={}),
    source = book,
    target = jesus,
    type='WRITTEN_BY',
    properties={}
)

written_by_jim = Relationship(
    source = book,
    target = jim,
    type='WRITTEN_BY',
    properties={}
)
if __name__ == "__main__":
    visualize_graph([book, jesus, jim], relationships)