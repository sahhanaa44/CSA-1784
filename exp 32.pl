% Pattern matching using unification
match(X, X). % succeeds when both arguments unify
% List head-tail pattern
first([H|_], H).
rest([_|T], T).
% Check membership via pattern
member(X, [X|_]).
member(X, [_|T]) :- member(X, T).
