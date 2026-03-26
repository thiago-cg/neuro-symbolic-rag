% ============================================================
% VFS Neuro-Symbolic — Base Datalog Rules
% ============================================================
% Facts are injected dynamically by clingo_engine.py
% from extracted triples. These rules define inferences.

% --- Direct influence (from any action predicate) ---
influencia_direta(X, Y) :- usa(X, Y).
influencia_direta(X, Y) :- cita(X, Y).
influencia_direta(X, Y) :- estende(X, Y).
influencia_direta(X, Y) :- supera(X, Y).
influencia_direta(X, Y) :- define(X, Y).

% --- Transitive influence ---
influencia(X, Y) :- influencia_direta(X, Y).
influencia(X, Z) :- influencia(X, Y), influencia_direta(Y, Z), X != Z.

% --- Comparability (two concepts influence the same target) ---
comparavel(X, Y) :- influencia(X, Z), influencia(Y, Z), X != Y.
comparavel(X, Y) :- comparavel(Y, X).

% --- Potential conflict (detected via contradiz predicate) ---
conflito_potencial(X, Y) :- contradiz(X, Y).
conflito_potencial(X, Y) :- contradiz(Y, X).

% --- Make all inferred atoms visible in the model ---
#show influencia_direta/2.
#show influencia/2.
#show comparavel/2.
#show conflito_potencial/2.
