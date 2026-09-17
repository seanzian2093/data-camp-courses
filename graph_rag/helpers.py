import os
import networkx as nx
import matplotlib.pyplot as plt

sizes = {'Book': 10000, 'Person': 5000, 'Chapter': 5000, 'Chunk': 2000, 'Entity': 1000}
# colours = {'Book': 'lightblue', 'Chapter': 'lightgreen', 'Chunk': 'lightpink', 'Entity': 'lightyellow'}
colours = {'Book': 'lightblue',
 'Chapter': 'lightgreen',
 'Chunk': 'lightpink',
 'Person': 'lightyellow'}

NEO4J_URL = os.getenv("NEO4J_URL", "neo4j://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Well-known authors mapped to a handful of their famous works
def visualize_graph(nodes = [], relationships = [], figsize=(8, 4)):
    node_sizes = [sizes[node.type] for node in nodes]
    node_colours = [colours[node.type] for node in nodes]
    
    G = nx.Graph()
    for node in nodes:
        G.add_node(node.id, **node.properties)
    for relationship in relationships:
        G.add_edge(relationship.source.id, relationship.target.id, **relationship.properties)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.margins(0.3) 

    nx.draw(G, with_labels=True,  font_weight='bold', ax=ax, arrows=True, arrowstyle='-|>', edge_color='navy',  font_size=6, node_size=node_sizes, node_color=node_colours),
    nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G), ax=ax, edge_labels=nx.get_edge_attributes(G, 'type'), font_size=6)
    
    plt.show()

    return fig, ax

visualize_node = lambda n: visualize_graph([n])