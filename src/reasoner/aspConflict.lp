% ─── ASP Conflict Resolution Rules ────────────────────────────────────────────
% Non-deterministic choice: for each conflict, choose which side to accept
{ accept(X) ; accept(Y) } = 1 :- conflict(X, Y).

% Evidence counting: weight by number of citations
evidence_count(X, N) :- accept(X), #count{ Z : cita(Z, X) } = N.

% Consensus: accepted item that is cited by the most works
consensus(X) :- accept(X), evidence_count(X, N),
                #max{ M : evidence_count(_, M) } = N.

% Integrity constraint: if X contradicts Y, at most one can be accepted
:- contradiz(X, Y), accept(X), accept(Y).

% Optimize: maximize total evidence for accepted items
#maximize { N,X : evidence_count(X, N) }.
