from collections import deque

def is_valid(m, c):
    if m < 0 or c < 0 or m > 3 or c > 3:
        return False
    if m > 0 and c > m:
        return False
    if (3 - m) > 0 and (3 - c) > (3 - m):
        return False
    return True

def solve():
    start = (3, 3, 1)
    goal = (0, 0, 0)

    moves = [(1,0),(2,0),(0,1),(0,2),(1,1)]

    q = deque([(start, [start])])
    visited = set()

    while q:
        state, path = q.popleft()

        if state == goal:
            for s in path:
                print(s)
            return

        m, c, boat = state

        for dm, dc in moves:
            if boat == 1:
                new = (m-dm, c-dc, 0)
            else:
                new = (m+dm, c+dc, 1)

            if is_valid(new[0], new[1]) and new not in visited:
                visited.add(new)
                q.append((new, path+[new]))

solve()
