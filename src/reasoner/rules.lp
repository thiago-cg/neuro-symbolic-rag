% ─── Datalog Base Rules ────────────────────────────────────────────────────────
% Transitive closure for citation chain
cita_indireta(X, Z) :- cita(X, Y), cita(Y, Z).
cita_indireta(X, Z) :- cita_indireta(X, Y), cita(Y, Z).

% Transitive closure for extension chain
estende_indiretamente(X, Z) :- estende(X, Y), estende(Y, Z).
estende_indiretamente(X, Z) :- estende_indiretamente(X, Y), estende(Y, Z).

% Comparability: two works can be compared if both treat the same concept
comparavel(X, Y) :- trata_conceito(X, C), trata_conceito(Y, C), X != Y.

% Inferred facts
inferred(cita_indireta(X, Z)) :- cita_indireta(X, Z).
inferred(estende_indiretamente(X, Z)) :- estende_indiretamente(X, Z).
inferred(comparavel(X, Y)) :- comparavel(X, Y).

% Conflict detection: direct contradiction
conflict(X, Y) :- contradiz(X, Y).
conflict(X, Y) :- contradiz(Y, X).

% Conflict via surpassing: if X supera Y and Y supera X
conflict(X, Y) :- supera(X, Y), supera(Y, X).
