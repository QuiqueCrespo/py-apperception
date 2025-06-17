#program step(t).
is_time(t).

#minimize {
              1@1,A,R : rule_body(R,A);
              1@1,A,R : rule_arrow_head(R,A);
              1@1,A,R : rule_causes_head(R,A);
              1@1,A : init(A);
              1@1,A : gen_permanent(A);
              1@1,A,T : force(A,T)
}.


%------------------------------------------------------------------------------
% evaluation
%------------------------------------------------------------------------------

eval_body(R, Subs, t) :-
    rule_subs(R, Subs),
    is_time(t),
    eval_atom(V, Subs, t) : rule_body(R, V).

eval_atom(VA, Subs, t) :-
    ground_atom(VA, GA, Subs),
    holds(GA, t).

eval_atom(VA, Subs, t) :-
    ground_atom(VA, GA, Subs),
    permanent(GA),
    is_time(t).

holds(s(C, Obj), t) :-
    init(s(C, Obj)),
    init_time(t).

holds(s2(C, Obj, Obj2), t) :-
    init(s2(C, Obj, Obj2)),
    init_time(t).
    
% frame axiom
holds(S, t) :-
    holds(S, t-1),
    is_time(t-1),
    is_time(t),
    not is_static_atom(S),
    not can_not_hold(S, t).

can_not_hold(S, t) :-
    holds(S2, t),
    incompossible(S, S2).

% causes update
holds(GC, t) :-     
    rule_causes_head(R, VC),
    eval_body(R, Subs, t-1),
    ground_atom(VC, GC, Subs),
    is_time(t),
    is_time(t-1).

% arrow update
holds(GA, t) :-
    rule_subs(R, Subs),
    rule_arrow_head(R, VA),
    ground_atom(VA, GA, Subs),
    eval_body(R, Subs, t).

% exogenous action
holds(GA, t) :- exogenous(GA, t), is_time(t).

%------------------------------------------------------------------------------
% Distinguished time-steps
%------------------------------------------------------------------------------

init_time(T) :- 
    is_time(T),    
    not is_time(T-1).

last_time(T) :- 
    is_time(T),    
    not is_time(T+1).

%------------------------------------------------------------------------------
% Sensations need to be connected to holds/2 atoms
% (Intuitions without concepts are blind)
%------------------------------------------------------------------------------

violation_kant_condition_blind_sense :-
    senses(S, T),
    is_time(T),
    not holds(S, T).

%------------------------------------------------------------------------------
% Aesthetic constraint: spatial unity
%------------------------------------------------------------------------------

violation_kant_condition_spatial_unity :- 
    spatial_unity_counterexample(X, Y, t).

spatial_unity_counterexample(X, Y, t) :-
    permanent(isa(t_object, X)), 
    permanent(isa(t_object, Y)), 
    is_time(t),
    not related(X, Y, t).

related(X, Y, t) :-
    holds(s2(_, X, Y), t).

related(X, Y, t) :-
    permanent(isa2(_, X, Y)),
    is_time(t).

related(X, X, t) :-
    permanent(isa(t_object, X)), 
    is_time(t).
    
related(X, Y, t) :- related(Y, X, t).

related(X, Y, t) :-
    related(X, Z, t),
    related(Z, Y, t).

%------------------------------------------------------------------------------
% Aesthetic constraint: part-of must create a labeled pointed graph
%------------------------------------------------------------------------------

violation_kant_condition_spatial_unity :- 
    interpretation_uses_part_of,
    not exists_spatial_unity.

exists_spatial_unity :-
    spatial_unifier(X).

spatial_unifier(X) :-    
    permanent(isa(t_object, X)),
    not spatial_unity_counterexample(X).

spatial_unity_counterexample(X) :-
    permanent(isa(t_object, X)),
    permanent(isa(t_object, Y)),
    X != Y,
    is_time(T),
    not part_of_trans(Y, X, T).

interpretation_uses_part_of :- part_of(_, _, _).

part_of_trans(Part, Whole, T) :-
    part_of(Part, Whole, T).

part_of_trans(X, Y, T) :-
    part_of(X, Z, T),
    part_of_trans(Z, Y, T).

part_of(Part, Whole, T) :-
    holds(s2(c_part, Part, Whole), T).

part_of(Part, Whole, T) :-
    is_time(T),
    permanent(isa2(p_part, Part, Whole)).

violation(spatial_cycle) :-
    part_of_trans(X, Y, T),
    part_of_trans(Y, X, T),
    X != Y.

%------------------------------------------------------------------------------
% Correctness
%------------------------------------------------------------------------------

incorrect :-
    hidden(A, T),
    not holds(A, T).

correct :- not incorrect.

%------------------------------------------------------------------------------
% Correctness of held-outs for pixel arrays
%------------------------------------------------------------------------------

incorrect :- incorrect_at(T).

incorrect_at(T) :- 
    test_sprite_at(T, C, S),
    not possible_sprite_at(T, C, S).

incorrect_at(T) :-
    impossible_combination_satisfied(T, N).

impossible_combination_satisfied(T, N) :-
    is_impossible_combination(T, N),
    not unsat(T, N).

unsat(T, N) :-
    test_impossible_combination(T, N, C, S),
    not possible_sprite_at(T, C, S).

is_impossible_combination(T, N) :- test_impossible_combination(T, N, _, _).

%%%

count_num_inaccurate(N) :- 
    N = #count { T : incorrect_at(T) }.

count_num_accurate(X-Y) :- 
    count_num_held_out_time_steps(X), count_num_inaccurate(Y).

count_num_held_out_time_steps(N) :- 
    N = #count { T : is_held_out_time_step(T) }.

is_held_out_time_step(T) :- 
    test_sprite_at(T, _, _).

%------------------------------------------------------------------------------
% Prevent general (non-Kantian) violations
%------------------------------------------------------------------------------

:- violation(_).
:- violation_kant_condition_blind_sense.
:- violation_kant_condition_spatial_unity.
:- incorrect.

