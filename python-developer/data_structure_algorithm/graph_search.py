import queue


def dfs(visited_vertices, graph, current_vertex):
    """Depth first search - graph"""
    # Check if the current vertex has not been visited
    if current_vertex not in visited_vertices:
        # Mark the current vertex as visited
        visited_vertices.add(current_vertex)
        # For each neighbour of the current vertex, call the dfs function
        for neighbour in graph[current_vertex]:
            dfs(visited_vertices, graph, neighbour)


def bfs(graph, initial_vertex):
    """Breadth first search - graph"""
    visited_vertices = []
    # A queue for loop
    bfs_queue = queue.SimpleQueue()
    bfs_queue.put(initial_vertex)
    # Visiting starts from initial_vertex
    visited_vertices.append(initial_vertex)
    while not bfs_queue.empty():
        current_vertex = bfs_queue.get()
        for adjacent_vertex in graph[current_vertex]:
            if adjacent_vertex not in visited_vertices:
                visited_vertices.append(adjacent_vertex)
                bfs_queue.put(adjacent_vertex)
    return visited_vertices


if __name__ == "__main__":
    graph = {
        "0": ["1", "2"],
        "1": ["0", "2", "3"],
        "2": ["0", "1", "4"],
        "3": ["1", "4"],
        "4": ["2", "3"],
    }

    dfs(set(), graph, "0")
    print(bfs(graph, "0"))
