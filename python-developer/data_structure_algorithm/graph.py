class Graph:
    def __init__(self) -> None:
        self.vertices = {}

    def add_vertex(self, vertex):
        self.vertices[vertex] = []

    def add_edge(self, source, target):
        self.vertices[source].append(target)


class WeightedGraph:
    def __init__(self) -> None:
        self.vertices = {}

    def add_vertex(self, vertex):
        self.vertices[vertex] = []

    def add_edge(self, source, target, weight):
        self.vertices[source].append([target, weight])


if __name__ == "__main__":
    my_graph = WeightedGraph()
    # Create vertices
    my_graph.add_vertex("Paris")
    my_graph.add_vertex("Toulouse")
    my_graph.add_vertex("Biarritz")
    # Create edges
    my_graph.add_edge("Paris", "Toulouse", 678)
    my_graph.add_edge("Toulouse", "Biarrita", 312)
    my_graph.add_edge("Biarrita", "Paris", 783)
