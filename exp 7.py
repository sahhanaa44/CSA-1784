from collections import deque

graph = {}

n = int(input("Enter number of nodes: "))

for i in range(n):
    node = input("Node: ")
    neighbors = input("Neighbors (space separated): ").split()
    graph[node] = neighbors

start = input("Enter starting node: ")

visited = set()
queue = deque([start])

print("BFS Traversal:")

while queue:
    node = queue.popleft()

    if node not in visited:
        print(node, end=" ")
        visited.add(node)

        for neighbor in graph[node]:
            queue.append(neighbor)
