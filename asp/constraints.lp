

%------------------------------------------------------------------------------
% Verify judgements.
%
% Every variable group referenced in a judgement must have a corresponding subs
%------------------------------------------------------------------------------

violation(var_group(R, VG)) :-
    rule_var_group(R, VG),
    not has_subs(VG).

has_subs(VG) :- subs_group(VG, _).

% Every variable that appears in a rule must be contained in the rule's group.
violation(undefined_var(R, V)) :-  
    var_in_rule(V, R), 
    rule_var_group(R, VG), 
    not contains_var(VG, V).

%------------------------------------------------------------------------------
% Ordering constraints (symmetry breaking)
%------------------------------------------------------------------------------

violation(bad_var_order(V1, V2, R)) :-
    less_var(V1, V2),
    var_in_body(V2, R),
    not appears_before(V1, V2, R).

violation(bad_rule_order(R1, R2)) :-
    earlier_rule(R1, R2),
    has_rule_body(R1),
    has_rule_body(R2),
    not less_rule(R1, R2).

% Symmetry-breaking constraint for use_rule
:-
    is_arrow_rule(R1),
    is_arrow_rule(R2),
    R1 < R2,
    use_rule(R2),
    not use_rule(R1).

% Symmetry-breaking constraint for use_rule
:-
    is_causes_rule(R1),
    is_causes_rule(R2),
    R1 < R2,
    use_rule(R2),
    not use_rule(R1).



%------------------------------------------------------------------------------
% Redundant constraints
%------------------------------------------------------------------------------

:-  init(A),
    init(B),
    incompossible(A, B).

:-  gen_permanent(A),
    gen_permanent(B),    
    incompossible(A, B).

:-  rule_body(R, A),
    rule_body(R, B),
    definitely_incompossible_var_atoms(A, B).    

:-  rule_body(R, A),
    rule_arrow_head(R, A).

:-  rule_body(R, A),
    rule_causes_head(R, A).

:-  rule_body(R, A),
    rule_arrow_head(R, B),
    definitely_incompossible_var_atoms(A, B).    

% Unsafe rules
:-  rule_arrow_head(R, s(_, V)),
    not var_in_body(V, R).

:-  rule_arrow_head(R, s2(_, V, _)),
    not var_in_body(V, R).

:-  rule_arrow_head(R, s2(_, _, V)),
    not var_in_body(V, R).

:-  rule_causes_head(R, s(_, V)),
    not var_in_body(V, R).

:-  rule_causes_head(R, s2(_, V, _)),
    not var_in_body(V, R).

:-  rule_causes_head(R, s2(_, _, V)),
    not var_in_body(V, R).



%------------------------------------------------------------------------------
% Prevent predicting exogenous 
%------------------------------------------------------------------------------


:-
    rule_arrow_head(_,s(A,_)),
    is_exogenous(A).

:-
    rule_causes_head(_,s(A,_)),
    is_exogenous(A).



%------------------------------------------------------------------------------
% Prevent predicting exogenous actions
%------------------------------------------------------------------------------