% ============================================================
% VFS Neuro-Symbolic — ASP Conflict Resolution Rules
% ============================================================
% Used in non-deterministic mode when conflito_potencial exists.
% Facts from rules.asp are loaded first, then these rules apply.

% --- Choice rule: choose exactly one position per conflict pair ---
1 { posicao_aceita(X) ; posicao_aceita(Y) } 1 :- conflito_potencial(X, Y).

% --- Constraint: cannot hold both sides of a contradiction ---
:- contradiz(X, Y), posicao_aceita(X), posicao_aceita(Y).

% --- Evidence: count papers citing each position ---
evidencia(X, N) :- posicao_aceita(X),
                   N = #count { P : cita(P, X) }.

% --- Consensus: positions cited by 3+ papers ---
consenso(X) :- posicao_aceita(X), evidencia(X, N), N >= 3.

% --- Show all relevant atoms ---
#show posicao_aceita/1.
#show evidencia/2.
#show consenso/1.
