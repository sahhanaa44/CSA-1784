graph = {
    'A': [('B',1), ('C',3)],
    'B': [('D',3), ('E',5)],
    'C': [('F',2)],
    'D': [],
    'E': [('G',2)],
    'F': [('G',1)],
    'G': []
}

heuristic = {
    'A': 7,
    'B': 6,
    'C': 4,
    'D': 5,
    'E': 2,
    'F': 1,
    'G': 0
}

open_list = [('A', 0)]
closed = []

goal = 'G'

while open_list:

    open_list.sort(key=lambda x: x[1])
    node, cost = open_list.pop(0)

    if node == goal:
        print("Goal Reached:", node)
        break

    closed.append(node)

    for neighbor, weight in graph[node]:

        if neighbor not in closed:
            f = cost + weight + heuristic[neighbor]
            open_list.append((neighbor, f))
