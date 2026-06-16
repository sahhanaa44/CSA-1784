is_vowel(a). is_vowel(e). is_vowel(i). is_vowel(o). is_vowel(u).
count_vowels([], 0).
count_vowels([H|T], Count) :-
is_vowel(H), !,
count_vowels(T, C1),
Count is C1 + 1.
count_vowels([_|T], Count) :-
count_vowels(T, Count).
