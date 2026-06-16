move(1, Source, Target, _) :-
write('Move top disk from '), write(Source),
write(' to '), write(Target), nl.
move(N, Source, Target, Aux) :-
N > 1,
N1 is N - 1,
move(N1, Source, Aux, Target),
move(1, Source, Target, Aux),
move(N1, Aux, Target, Source).
