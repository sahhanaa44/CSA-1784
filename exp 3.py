def water_jug(x, y):
    visited = set()

    def dfs(a, b, path):
        if (a, b) in visited:
            return False

        visited.add((a, b))
        path.append((a, b))

        # Goal State
        if (a, b) == (2, 0):
            print("Solution Path:")
            for state in path:
                print(state)
            return True

        next_states = [
            (4, b),  # Fill 4L jug
            (a, 3),  # Fill 3L jug
            (0, b),  # Empty 4L jug
            (a, 0),  # Empty 3L jug

            # Pour 4L -> 3L
            (max(0, a - (3 - b)),
             min(3, a + b)),

            # Pour 3L -> 4L
            (min(4, a + b),
             max(0, b - (4 - a)))
        ]

        for na, nb in next_states:
            if dfs(na, nb, path.copy()):
                return True

        return False

    dfs(x, y, [])

# Start State
water_jug(0, 0)
