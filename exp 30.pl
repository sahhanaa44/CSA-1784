% Facts
has_wings. cannot_fly.
can_fly :- has_wings.
bird(penguin) :- cannot_fly.
bird(eagle) :- can_fly.
% Backward chaining rules
infer(X) :- bird(X).
infer(X) :- can_fly, X = eagle.
infer(X) :- cannot_fly, X = penguin.
