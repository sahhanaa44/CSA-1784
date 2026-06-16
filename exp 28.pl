% Symptom predicate (simplified for demonstration)
has_symptom(fever). has_symptom(cough). has_symptom(rash).
has_symptom(conjunctivitis). has_symptom(runny_nose).
hypothesis(measles) :-
has_symptom(fever), has_symptom(cough),
has_symptom(conjunctivitis), has_symptom(runny_nose),
has_symptom(rash).
hypothesis(flu) :-
has_symptom(fever), has_symptom(cough), has_symptom(runny_nose).
hypothesis(common_cold) :-
has_symptom(runny_nose), has_symptom(cough).
diagnose(Disease) :- hypothesis(Disease).
