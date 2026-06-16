% Graph edges
edge(0,1,3).
edge(0,2,6).
edge(1,3,5).
edge(1,4,9).
edge(2,5,8).
edge(3,6,4).
edge(4,7,2).
edge(5,8,7).
edge(6,9,1).

% Heuristic values
h(0,10).
h(1,8).
h(2,7).
h(3,6).
h(4,5).
h(5,4).
h(6,3).
h(7,2).
h(8,1).
h(9,0).

best_first(Start, Goal) :-
    search([[Start]], Goal).

search([[Goal|Rest]|_], Goal) :-
    reverse([Goal|Rest], Path),
    write('Path: '),
    write(Path), nl.

search([Path|Paths], Goal) :-
    extend(Path, NewPaths),
    append(Paths, NewPaths, Temp),
    sort_paths(Temp, Sorted),
    search(Sorted, Goal).

extend([Node|Rest], NewPaths) :-
    findall(
        [Next,Node|Rest],
        (edge(Node,Next,_),
         \+ member(Next,[Node|Rest])),
        NewPaths
    ).

sort_paths(Paths, Sorted) :-
    predsort(compare_paths, Paths, Sorted).

compare_paths(<, [A|_], [B|_]) :-
    h(A, HA),
    h(B, HB),
    HA < HB.

compare_paths(>, [A|_], [B|_]) :-
    h(A, HA),
    h(B, HB),
    HA >= HB.
