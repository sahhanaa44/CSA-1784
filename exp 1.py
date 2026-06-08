import copy
from heapq import heappush, heappop

n = 3

# bottom, left, top, right
row = [1, 0, -1, 0]
col = [0, -1, 0, 1]


# A class for Priority Queue
class priorityQueue:

    def __init__(self):
        self.heap = []

    # Inserts a new key 'k'
    def push(self, k):
        heappush(self.heap, k)

    # Method to remove minimum element
    def pop(self):
        return heappop(self.heap)

    # Method to know if the Queue is empty
    def empty(self):
        return not self.heap


# Node structure
class node:

    def __init__(self, parent, mat, empty_tile_pos, cost, level):
        self.parent = parent
        self.mat = mat
        self.empty_tile_pos = empty_tile_pos
        self.cost = cost
        self.level = level

    # Priority based on cost
    def __lt__(self, nxt):
        return self.cost < nxt.cost


# Function to calculate the number of misplaced tiles
def calculateCost(mat, final):
    count = 0

    for i in range(n):
        for j in range(n):
            if mat[i][j] and mat[i][j] != final[i][j]:
                count += 1

    return count


def newNode(mat, empty_tile_pos, new_empty_tile_pos,
            level, parent, final):

    # Copy parent matrix
    new_mat = copy.deepcopy(mat)

    # Move tile
    x1, y1 = empty_tile_pos
    x2, y2 = new_empty_tile_pos

    new_mat[x1][y1], new_mat[x2][y2] = (
        new_mat[x2][y2],
        new_mat[x1][y1]
    )

    cost = calculateCost(new_mat, final)

    return node(
        parent,
        new_mat,
        new_empty_tile_pos,
        cost,
        level
    )


# Function to print matrix
def printMatrix(mat):
    for i in range(n):
        for j in range(n):
            print("%d" % mat[i][j], end=" ")
        print()


# Check valid coordinates
def isSafe(x, y):
    return 0 <= x < n and 0 <= y < n


# Print path from root to destination
def printPath(root):
    if root is None:
        return

    printPath(root.parent)
    printMatrix(root.mat)
    print()


# Function to solve puzzle using Branch and Bound
def solve(initial, empty_tile_pos, final):

    pq = priorityQueue()

    cost = calculateCost(initial, final)

    root = node(
        None,
        initial,
        empty_tile_pos,
        cost,
        0
    )

    pq.push(root)

    while not pq.empty():

        minimum = pq.pop()

        if minimum.cost == 0:
            printPath(minimum)
            return

        for i in range(4):

            new_tile_pos = [
                minimum.empty_tile_pos[0] + row[i],
                minimum.empty_tile_pos[1] + col[i]
            ]

            if isSafe(new_tile_pos[0], new_tile_pos[1]):

                child = newNode(
                    minimum.mat,
                    minimum.empty_tile_pos,
                    new_tile_pos,
                    minimum.level + 1,
                    minimum,
                    final
                )

                pq.push(child)


# Driver Code

# Initial configuration
initial = [
    [1, 2, 3],
    [5, 6, 0],
    [7, 8, 4]
]

print("Initial:")
for i in initial:
    print(i)

# Final configuration
final = [
    [1, 2, 3],
    [5, 8, 6],
    [0, 7, 4]
]

print("Final:")
for i in final:
    print(i)

# Blank tile coordinates
empty_tile_pos = [1, 2]

print("Operations:")

# Solve puzzle
solve(initial, empty_tile_pos, final)
