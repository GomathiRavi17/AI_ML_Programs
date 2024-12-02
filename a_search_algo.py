from heapq import heappop, heappush

def a_star_search(graph: dict, start: str, goal: str, heuristic_values: dict) -> (int, list):
    '''
    A* search algorithm implementation.

    @param graph: The graph to search.
    @param start: The starting node.
    @param goal: The goal node.
    @param heuristic_values: The heuristic values for each node. The goal node must be admissible, and the heuristic value must be 0.
    @return: A tuple containing the path cost from the start node to the goal node and the optimal path as a list of nodes.
    '''

    # A min heap is used to implement the priority queue for the open list.
    open_list = [(heuristic_values[start], start, [start])]  # Each entry: (cost, current node, path to the node)
    closed_list = set()

    while open_list:
        cost, node, path = heappop(open_list)

        # If the goal node is reached, return the cost and the path.
        if node == goal:
            return cost, path

        if node in closed_list:
            continue

        closed_list.add(node)

        # Subtract the heuristic value as it was overcounted.
        current_path_cost = cost - heuristic_values[node]

        for neighbor, edge_cost in graph[node]:
            if neighbor in closed_list:
                continue

            # f(x) = g(x) + h(x), where g(x) is the path cost and h(x) is the heuristic.
            neighbor_cost = current_path_cost + edge_cost + heuristic_values[neighbor]
            heappush(open_list, (neighbor_cost, neighbor, path + [neighbor]))

    return -1, []  # No path found

# Example graph and heuristic values
EXAMPLE_GRAPH = {
    'a': [('b', 9), ('c', 4), ('d', 7)],
    'b': [('e', 11)],
    'c': [('e', 17), ('f', 12)],
    'd': [('f', 14)],
    'e': [('z', 5)],
    'f': [('z', 9)]
}

EXAMPLE_HEURISTIC_VALUES = {
    'a': 21,
    'b': 14,
    'c': 18,
    'd': 18,
    'e': 5,
    'f': 8,
    'z': 0
}

# Run the A* search and print the results
EXAMPLE_RESULT = a_star_search(EXAMPLE_GRAPH, 'a', 'z', EXAMPLE_HEURISTIC_VALUES)
print(f"Optimal Cost: {EXAMPLE_RESULT[0]}")
print(f"Optimal Path: {' -> '.join(EXAMPLE_RESULT[1])}")