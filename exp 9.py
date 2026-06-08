from itertools import permutations

n = int(input("Enter number of cities: "))

cost = []

print("Enter Cost Matrix:")

for i in range(n):
    row = list(map(int, input().split()))
    cost.append(row)

cities = list(range(n))

min_cost = float('inf')

for path in permutations(cities[1:]):
    tour = [0] + list(path) + [0]

    total = 0

    for i in range(len(tour)-1):
        total += cost[tour[i]][tour[i+1]]

    min_cost = min(min_cost, total)

print("Minimum Cost =", min_cost)
