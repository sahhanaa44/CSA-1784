# Map Coloring using CSP

states = ['A', 'B', 'C', 'D']

neighbors = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['B', 'C']
}

colors = ['Red', 'Green', 'Blue']

solution = {}

def is_safe(state, color):
    for neighbor in neighbors[state]:
        if neighbor in solution and solution[neighbor] == color:
            return False
    return True

def map_coloring(index):
    if index == len(states):
        return True

    state = states[index]

    for color in colors:
        if is_safe(state, color):
            solution[state] = color

            if map_coloring(index + 1):
                return True

            del solution[state]

    return False

if map_coloring(0):
    print("Color Assignment:")
    for state, color in solution.items():
        print(state, "->", color)
else:
    print("No solution found")
