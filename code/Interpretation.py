# interpretation.py

from dataclasses import dataclass, field, InitVar
from enum import Enum
from typing import List, Tuple, Optional, Dict, Union, Any
import re
from collections import defaultdict
from enum import Enum
import math



from itertools import combinations
import os


# -------------------------------------- Flags ----------------------------------


flag_ablation_remove_kant_condition_blind_sense: bool = False
flag_ablation_remove_kant_condition_spatial_unity: bool = False
flag_ablation_remove_kant_condition_conceptual_unity: bool = False

flag_unicode: bool = False




# Global divider used in generated files
divider = "%------------------------------------------------------------------------------"




# Simple aliases
Atom = str
RuleID = str


# Set exists_string based on flag_unicode if available
exists_string = "∃" if False else "exists"

# Utility strings based on flags
def and_string() -> str:
    return " ∧ " if flag_unicode else " /\\ "

def arrow_string() -> str:
    return " → " if flag_unicode else " -> "

def causes_string() -> str:
    return " ▸ " if flag_unicode else " >> "

def xor_string() -> str:
    return " ⊕ " if flag_unicode else " + "


# Basic nameable entities with string representations
@dataclass(frozen=True, order=True)
class Type:
    name: str
    def __str__(self) -> str:
        return f"t_{self.name}"

@dataclass(frozen=True, order=True)
class Concept:
    name: str

@dataclass(frozen=True, order=True)
class C(Concept):
    def __str__(self) -> str:
        return f"c_{self.name}"

@dataclass(frozen=True, order=True)
class P(Concept):
    def __str__(self) -> str:
        return f"p_{self.name}"

@dataclass(frozen=True, order=True)
class Object:
    name: str
    def __str__(self) -> str:
        return f"obj_{self.name}"

@dataclass(frozen=True, order=True)
class Var:
    name: str
    def __str__(self) -> str:
        return f"var_{self.name}"

# Predicate type enumeration
class PredicateType(Enum):
    FLUENT = "fluent"
    PERMANENT = "permanent"
    def __str__(self) -> str:
        return self.value
    
Subs = List[Tuple[Var, Object]]
SubsGroup = Tuple[str, List[Subs]]

# Lineage for concepts
class ConceptLineage(Enum):
    GIVEN = "Given"
    CONSTRUCTED = "Constructed"

# Ground (instantiated) atoms
@dataclass(frozen=True)
class GroundAtom:
    concept: Concept
    objects: List[Object]

@dataclass(frozen=True)
class GA(GroundAtom):  # Fluent initialization
    def __str__(self) -> str:
        objs = self.objects
        if len(objs) == 1:
            return f"init(s({self.concept}, {objs[0]}))"
        elif len(objs) == 2:
            return f"init(s2({self.concept}, {objs[0]}, {objs[1]}))"
        else:
            raise ValueError(f"GA expects 1 or 2 objects, got {len(objs)}")

@dataclass(frozen=True)
class Perm(GroundAtom):  # Permanent facts
    def __str__(self) -> str:
        objs = self.objects
        if len(objs) == 1:
            return f"gen_permanent(isa({self.concept}, {objs[0]}))"
        elif len(objs) == 2:
            return f"gen_permanent(isa2({self.concept}, {objs[0]}, {objs[1]}))"
        else:
            raise ValueError(f"Perm expects 1 or 2 objects, got {len(objs)}")

# Variable-based atoms for rules
@dataclass(frozen=True)
class VarAtom:
    pass

@dataclass(frozen=True)
class VA(VarAtom):  # Fluent variable condition
    concept: Concept
    vars: List[Var]
    def __str__(self) -> str:
        vs = self.vars
        if len(vs) == 1:
            return f"s({self.concept}, {vs[0]})"
        elif len(vs) == 2:
            return f"s2({self.concept}, {vs[0]}, {vs[1]})"
        else:
            raise ValueError(f"VA expects 1 or 2 vars, got {len(vs)}")

@dataclass(frozen=True)
class Isa(VarAtom):  # Type assertion
    concept: Concept
    var: Var
    def __str__(self) -> str:
        return f"isa({self.concept}, {self.var})"

@dataclass(frozen=True)
class Isa2(VarAtom):  # Binary type assertion
    concept: Concept
    var1: Var
    var2: Var
    def __str__(self) -> str:
        return f"isa2({self.concept}, {self.var1}, {self.var2})"

# Logical rule classes
@dataclass(frozen=True)
class Rule:
    id: str

    def parse_rule_atom(self, atom: str) -> Union[str]:  # Replace str with GroundAtom or VarAtom if available
        """
        Parses a rule atom string and returns the corresponding GroundAtom or VarAtom.
        Capitalizes predicate and argument names.
        """
        match = re.fullmatch(r"\s*(\w+)\s*\(([^)]*)\)\s*", atom)
        if not match:
            raise ValueError(f"Invalid rule atom format: {atom}")

        _, atom_ = match.groups()
        pred = "_".join(atom_.strip().split(",")[0].split("_")[1:])
        args = ["_".join(arg.split("_")[1:]).strip().capitalize() for arg in atom_.split(",")[1:] if arg.strip()]

        return f"{pred}({', '.join(args)})"



        
        

@dataclass(frozen=True)
class Arrow(Rule):
    body: List[str]
    head: str
    def __str__(self) -> str:
        return f"{self.id} : {and_string().join([self.parse_rule_atom(a) for a in self.body])}{arrow_string()}{self.parse_rule_atom(self.head)}"

@dataclass(frozen=True)
class Causes(Rule):
    body: List[str]
    head: str
    def __str__(self) -> str:
        return f"{self.id} : {and_string().join([self.parse_rule_atom(a) for a in self.body])}{causes_string()}{self.parse_rule_atom(self.head)}"


@dataclass(frozen=True)
class Xor(Rule):
    body: List[str]
    heads: List[str]
    def __str__(self) -> str:
        return f"{self.id} : {and_string().join([self.parse_rule_atom(a) for a in self.body])}{arrow_string()}{xor_string().join(self.parse_rule_atom(self.head))}"

@dataclass
class InterpretationStatistics:
    num_used_arrow_rules: int
    num_used_causes_rules: int
    total_body_atoms: int
    num_inits: int
    bnn_entropy: Optional[float]
    ambiguity: Optional[int]
    possible_preds: List[str]
    def total_cost(self) -> int:
        return self.num_used_arrow_rules + self.num_used_causes_rules + self.total_body_atoms + self.num_inits
    def total_num_clauses(self) -> int:
        return self.num_used_arrow_rules + self.num_used_causes_rules + self.num_inits

@dataclass
class Interpretation:
    xs: InitVar[List[str]]

    times: List[int]                      = field(init=False)
    senses: List[Atom]                    = field(init=False)
    hiddens: List[Atom]                   = field(init=False)
    exclusions: List[str]                 = field(init=False)
    inits: List[Atom]                     = field(init=False)
    permanents: List[Atom]                = field(init=False)
    rules: List[Rule]                     = field(init=False)
    facts: List[Tuple[int, List[Atom]]]   = field(init=False)
    forces: List[Tuple[int, List[Atom]]]  = field(init=False)
    correct: bool                         = field(init=False)
    num_accurate: Optional[int]           = field(init=False)
    num_held_outs: Optional[int]          = field(init=False)
    statistics: InterpretationStatistics  = field(init=False)

    def __post_init__(self, xs: List[str]) -> None:
        self.times         = [int(t) for t in self.extract_atoms("is_time(", xs)]
        self.senses        = self.extract_atoms("senses(", xs)
        self.hiddens       = self.extract_atoms("hidden(", xs)
        self.exclusions    = self.extract_exclusions(xs)
        self.inits         = self.extract_atoms("init(", xs)
        self.permanents    = self.extract_atoms("gen_permanent(", xs)
        self.rules         = self.extract_rules(xs)
        self.facts         = self.extract_facts(xs)
        self.forces        = self.extract_forces(xs)
        self.correct       = self.extract_correct(xs)
        self.num_accurate  = self.extract_maybe_int("count_num_accurate(", xs)
        self.num_held_outs = self.extract_maybe_int("count_num_held_out_time_steps(", xs)
        self.statistics    = self.extract_statistics(xs)

    # --- rule extraction ---
    def extract_rules(self, xs: List[str]) -> List[Any]:
        return self.extract_xors(xs) + self.extract_arrows(xs) + self.extract_causes(xs)

    def extract_xors(self, xs: List[str]) -> List[Xor]:
        heads = self.extract_xor_heads(xs)
        return [Xor(r, self.extract_body(xs, r), hs)
                for r, hs in heads
                if not r.startswith("r_input")]

    def extract_arrows(self, xs: List[str]) -> List[Arrow]:
        return [Arrow(r, self.extract_body(xs, r), h)
                for r, h in self.extract_arrow_heads(xs)]

    def extract_causes(self, xs: List[str]) -> List[Causes]:
        return [Causes(r, self.extract_body(xs, r), h)
                for r, h in self.extract_cause_heads(xs)]

    def extract_xor_heads(self, xs: List[str]) -> List[Tuple[str, List[str]]]:
        ps: List[Tuple[str,List[str]]] = []
        for x in xs:
            if x.startswith("rule_head_xor("):
                body  = x[len("rule_head_xor("):-1]
                parts = self.bimble_split(body, ',')
                ps.append((parts[0], parts[1:]))
        return self.collect_pairs(ps)

    def extract_arrow_heads(self, xs: List[str]) -> List[Tuple[str, str]]:
        out: List[Tuple[str,str]] = []
        for x in xs:
            if x.startswith("rule_arrow_head("):
                body = x[len("rule_arrow_head("):-1]
                out.append(self.extract_arrow_pair(body))
        return out

    def extract_cause_heads(self, xs: List[str]) -> List[Tuple[str, str]]:
        out: List[Tuple[str,str]] = []
        for x in xs:
            if x.startswith("rule_causes_head("):
                body = x[len("rule_causes_head("):-1]
                out.append(self.extract_cause_pair(body))
        return out

    def collect_pairs(self, ps: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
        d: Dict[str, List[str]] = defaultdict(list)
        for key, vals in ps:
            d[key].extend(vals)
        return list(d.items())

    def extract_body(self, xs: List[str], head: str) -> List[str]:
        prefix = f"rule_body({head},"
        return [x[len(prefix):-1] for x in xs if x.startswith(prefix)]

    def extract_arrow_pair(self, s: str) -> Tuple[str, str]:
        parts = s.split(",")
        return parts[0], ", ".join(parts[1:])

    def extract_cause_pair(self, s: str) -> Tuple[str, str]:
        parts = s.split(",")
        return parts[0], ", ".join(parts[1:])

    # --- facts & forces ---
    def extract_facts(self, xs: List[str]) -> List[Tuple[int, List[str]]]:
        return self.bring_together(self.extract_pred("holds(", xs))

    def extract_forces(self, xs: List[str]) -> List[Tuple[int, List[str]]]:
        return self.bring_together(self.extract_pred("force(", xs))

    def extract_pred(self, p: str, xs: List[str]) -> List[Tuple[int, str]]:
        out: List[Tuple[int,str]] = []
        for x in xs:
            if x.startswith(p):
                body = x[len(p):-1]
                parts = body.split(',')
                t = int(parts[-1])
                pred = parts[0].split("(")[1].split("_")[1:]  # Remove 'c_' or 's_' prefix
                pred = "_".join(pred).strip()  # Capitalize predicate
                args = ["_".join(arg.split('_')[1:]).strip() for arg in parts[1:-1] if arg.strip()]
                a = f"{pred}({', '.join(args)}"
                out.append((t, a))
        return out

    def bring_together(self, xs: List[Tuple[int, str]]) -> List[Tuple[int, List[str]]]:
        d: Dict[int, List[str]] = defaultdict(list)
        for t, atom in xs:
            d[t].append(atom)
        return [(t, d[t]) for t in sorted(d)]

    # --- exclusions & correctness ---
    def extract_exclusions(self, xs: List[str]) -> List[str]:
        p = 'exclusion_output("'
        out = []
        for x in xs:
            if x.startswith(p):
                out.append(x[len(p):-2])
        return out

    def extract_correct(self, xs: List[str]) -> bool:
        return any(x.startswith("correct") for x in xs)

    # --- integer & atom extraction ---
    def extract_maybe_int(self, p: str, xs: List[str]) -> Optional[int]:
        vals = self.extract_atoms(p, xs)
        if not vals:
            return None
        if len(vals) == 1:
            return int(vals[0])
        raise ValueError(f"Multiple matches for {p}")

    def extract_atoms(self, p: str, xs: List[str]) -> List[str]:
        atoms = [x[len(p):-1] for x in xs if x.startswith(p)]
        cleaned_atoms = []
        for atom in atoms:
            atom = atom.split(',')
            pred = "_".join(atom[0].strip().split('_')[1:])  # Remove 'c_' or 's_' prefix
            args = [
                "_".join(arg.split('_')[1:]).strip()# Capitalize and remove prefix
                for arg in atom[1:] if arg.strip()
            ]
            cleaned_atoms.append(f"{pred}({', '.join(args)}")
        return cleaned_atoms
    
        

    # --- statistics ---
    def extract_num_used_arrow_rules(self, xs: List[str]) -> int:
        return len(self.extract_atoms("used_arrow_rule(", xs))

    def extract_num_used_causes_rules(self, xs: List[str]) -> int:
        return len(self.extract_atoms("used_causes_rule(", xs))

    def extract_total_body_atoms(self, xs: List[str]) -> int:
        return len(self.extract_atoms("rule_body(", xs))

    def extract_num_inits(self, xs: List[str]) -> int:
        return len(self.extract_atoms("init(", xs))

    def extract_num_gen_permanents(self, xs: List[str]) -> int:
        return len(self.extract_atoms("gen_permanent(", xs))

    def extract_bnn_entropy(self, xs: List[str]) -> Optional[float]:
        bnn_es = self.extract_atoms("count_bnn_examples_per_predicate(", xs)
        pps    = self.extract_atoms("is_possible_pred(", xs)
        bvs    = self.extract_atoms("num_bvs(", xs)
        if not bnn_es:
            return None
        total = int(bvs[0]) if bvs else 0
        if total == 0 or len(pps) <= 1:
            return 0.0
        counts = [int(s.split(",")[1]) for s in bnn_es]
        freqs  = [c/total for c in counts if total>0]
        base   = len(pps)
        return sum(-p * math.log(p, base) for p in freqs if p > 0)

    def extract_ambiguity(self, xs: List[str]) -> Optional[int]:
        pairs = [tuple(s.split(",")) for s in self.extract_atoms("possible_pred(", xs)]
        if not pairs:
            return None
        m: Dict[str, List[str]] = defaultdict(list)
        for a, b in pairs:
            m[a].append(b)
        return sum(len(v) - 1 for v in m.values())

    def extract_possible_preds(self, xs: List[str]) -> List[str]:
        return self.extract_atoms("possible_pred(", xs)

    def extract_statistics(self, xs: List[str]) -> InterpretationStatistics:
        return InterpretationStatistics(
            num_used_arrow_rules   = self.extract_num_used_arrow_rules(xs),
            num_used_causes_rules  = self.extract_num_used_causes_rules(xs),
            total_body_atoms       = self.extract_total_body_atoms(xs),
            num_inits              = self.extract_num_inits(xs) + self.extract_num_gen_permanents(xs),
            bnn_entropy            = self.extract_bnn_entropy(xs),
            ambiguity              = self.extract_ambiguity(xs),
            possible_preds         = self.extract_possible_preds(xs),
        )

class PredicateType(Enum):
    IS_FLUENT = "IsFluent"
    IS_PERMANENT = "IsPermanent"

@dataclass
class Frame:
    types: List[Type] = field(default_factory=list)
    type_hierarchy: List[Tuple[Type, List[Type]]] = field(default_factory=list)
    objects: List[Tuple[Object, Type]] = field(default_factory=list)
    exogeneous_objects: List[Object] = field(default_factory=list)
    permanent_concepts: List[Tuple[Concept, ConceptLineage, List[Type]]] = field(default_factory=list)
    fluid_concepts: List[Tuple[Concept, List[Type]]] = field(default_factory=list)
    input_concepts: List[Concept] = field(default_factory=list)
    static_concepts: List[Concept] = field(default_factory=list)
    vars: List[Tuple[Var, Type]] = field(default_factory=list)
    var_groups: List[List[Var]] = field(default_factory=list)
    aux_files: List[str] = field(default_factory=list)

    def gen_elements(self) -> List[str]:
        xs: List[str] = []
        for c, _ in self.fluid_concepts:
            xs.append(f"is_concept({c}).")
        for c, _, _ in self.permanent_concepts:
            xs.append(f"is_concept({c}).")
        for c in self.static_concepts:
            xs.append(f"is_static_concept({c}).")
        for t in self.types:
            xs.append(f"is_type({t}).")
        return [divider, "% Elements", divider, ""] + xs + [""]

    def get_binary_concepts(self,
                            cs: List[Tuple[Concept, List[Type]]]
                           ) -> List[Tuple[Concept, List[Type]]]:
        return [
            (c, ts)
            for c, ts in cs
            if len(ts) == 2 and c not in self.input_concepts
        ]

    def gen_unary_concepts(self) -> Dict[Type, List[Concept]]:
        acc: Dict[Type, List[Concept]] = {}
        for c, ts in self.fluid_concepts:
            if len(ts) == 1 and c not in self.input_concepts:
                acc.setdefault(ts[0], []).append(c)
        return acc

    def gen_permanent_concepts(self) -> Dict[Type, List[Concept]]:
        acc: Dict[Type, List[Concept]] = {}
        for c, lineage, ts in self.permanent_concepts:
            if lineage is ConceptLineage.CONSTRUCTED and len(ts) == 1:
                acc.setdefault(ts[0], []).append(c)
        return acc

    def get_permanent_constructed_concepts(self) -> List[Tuple[Concept, List[Type]]]:
        return [
            (c, ts)
            for c, lineage, ts in self.permanent_concepts
            if lineage is ConceptLineage.CONSTRUCTED
        ]

    def var_group_for_type(self, t: Type) -> str:
        v = self.var_for_type(t)
        return f"var_group_{v.name}"

    def var_for_type(self, t: Type) -> Var:
        for v, ty in self.vars:
            if ty == t:
                return v
        raise ValueError(f"var_for_type: Type {t} not found")

    def var_group_name(self, group: List[Var]) -> str:
        names = sorted(v.name for v in group)
        return "_".join(names)

    def print_var_fluent(self, file: str, a: VarAtom) -> None:
        for vg in self.all_var_groups(a):
            with open(file, "a") as fh:
                fh.write(f"var_fluent({a}, var_group_{self.var_group_name(vg)}).\n")

    def print_var_isa(self, file: str, a: VarAtom) -> None:
        for vg in self.all_var_groups(a):
            with open(file, "a") as fh:
                fh.write(f"var_permanent({a}, var_group_{self.var_group_name(vg)}).\n")

    def all_var_groups(self, a: VarAtom) -> List[List[Var]]:
        if isinstance(a, VA):
            return [vg for vg in self.var_groups if set(a.vars).issubset(vg)]
        if isinstance(a, Isa):
            return [vg for vg in self.var_groups if a.var in vg]
        if isinstance(a, Isa2):
            return [vg for vg in self.var_groups if a.var1 in vg and a.var2 in vg]
        return []

    def subset(self, a: List[Any], b: List[Any]) -> bool:
        return all(elem in b for elem in a)

    def sub_types(self) -> List[Tuple[Type, Type]]:
        result: List[Tuple[Type, Type]] = []
        for super_t, subs in self.type_hierarchy:
            for sub_t in subs:
                result.append((sub_t, super_t))
        return result

    def sub_types_star(self) -> List[Tuple[Type, Type]]:
        st = set(self.sub_types() + [(t, t) for t in self.types])
        changed = True
        while changed:
            changed = False
            for a, b in list(st):
                for c, d in list(st):
                    if b == c and (a, d) not in st:
                        st.add((a, d))
                        changed = True
        return list(st)

    # ------------------------------- all_var_atoms ---------------------------------

    def all_var_fluents(self) -> List[VarAtom]:
        result: List[VarAtom] = []
        for c, ts in self.fluid_concepts:
            for vs in self.all_var_tuples(ts):
                result.append(VA(c, vs))
        return result

    def all_var_tuples(self, types_list: List[Type]) -> List[List[Var]]:
        if not types_list:
            return [[]]
        first, *rest = types_list
        sts = self.sub_types_star()
        vs_list = [v for v, t in self.vars if (t, first) in sts]
        result: List[List[Var]] = []
        for v in vs_list:
            for xs in self.all_var_tuples(rest):
                result.append([v] + xs)
        return result

    def all_var_isas(self) -> List[VarAtom]:
        result: List[VarAtom] = []
        sts = self.sub_types_star()

        for c, lineage, types in self.permanent_concepts:
            if len(types) == 1:
                (t1,) = types
                for v1, t1p in self.vars:
                    if (t1p, t1) in sts:
                        result.append(Isa(c, v1))
            elif len(types) == 2:
                t1, t2 = types
                for v1, t1p in self.vars:
                    if (t1p, t1) in sts:
                        for v2, t2p in self.vars:
                            if (t2p, t2) in sts:
                                result.append(Isa2(c, v1, v2))
        return result

    def xor_groups(self) -> List[List[str]]:
        groups: List[List[str]] = []
        # Fluent unary
        for t, cs in self.gen_unary_concepts().items():
            groups.append(self.xor_group(PredicateType.IS_FLUENT, t, sorted(cs)))
        # Permanent unary
        for t, cs in self.gen_permanent_concepts().items():
            groups.append(self.xor_group(PredicateType.IS_PERMANENT, t, sorted(cs)))
        return groups

    def num_xor_groups(self) -> int:
        return sum(len(grp) for grp in self.xor_groups())

    def xor_groups_strings(self) -> List[str]:
        lines: List[str] = []
        lines.append(f"#const k_xor_group=1.")
        lines.append("")
        lines.append("xor_group(k_xor_group).")
        lines.append("")
        xs = self.xor_groups()
        for i, grp in enumerate(xs, start=1):
            for predicate in grp:
                lines.append(f"{predicate} :- xor_group({i}).")
            lines.append("")
        return lines



    def gen_exists_constraints(self) -> List[str]:
        lines: List[str] = []
        print("Warning: the exist constrain implementation may not correct")
        # Fluid binary
        for c, ts in self.get_binary_concepts(self.fluid_concepts):
            print(f"gen_exists_constraints: {c} {ts}")
            
            lines.extend(self.gen_exists_constraints_for_pred(PredicateType.IS_FLUENT, (c, ts)))
        # Permanent binary
        print("gen_exists_constraints: permanent")
        for c, ts in self.get_binary_concepts(self.get_permanent_constructed_concepts()):

            print(f"gen_exists_constraints: {c} {ts}")
            lines.extend(self.gen_exists_constraints_for_pred(PredicateType.IS_PERMANENT, (c, ts)))
        return lines

    def gen_exists_constraints_for_pred(self,
                                       k: PredicateType,
                                       pair: Tuple[Concept, List[Type]]
                                      ) -> List[str]:
        c, ts = pair
        lines: List[str] = []
        lines.append("% Concept hierarchy")
        lines.append(f"sub_concept({c}, {ts[0]}).")
        lines.append("")
        lines.extend(self.gen_exists_at_most(k, (c, ts)))
        lines.extend(self.gen_exists_at_least(k, (c, ts)))
        lines.extend(self.gen_exists_incompossible(k, (c, ts)))
        return lines

    def gen_exists_at_most(self,
                           k: PredicateType,
                           pair: Tuple[Concept, List[Type]]
                          ) -> List[str]:
        c, _ = pair
        return [
            f"% {exists_string}! clause for {c} : at most one",
            ":-",
            f"\t{self.gen_atom2(k, c, 'X', 'Y')},",
            f"\t{self.gen_atom2(k, c, 'X', 'Y2')},",
            "\tY != Y2.",
            ""
        ]

    def gen_exists_at_least(self,
                            k: PredicateType,
                            pair: Tuple[Concept, List[Type]]
                           ) -> List[str]:
        c, ts = pair
        lines: List[str] = []
        lines.append(f"% {exists_string}! clause for {c} : at least one")
        lines.append(":-")
        lines.append(f"\tpermanent(isa({ts[0]}, X)),")
        lines.append("\tis_time(T),")
        lines.append(f"\tnot aux_{c}(X, T).")
        lines.append("")
        lines.append(f"aux_{c}(X, T) :-")
        lines.append("\tis_time(T),")
        lines.append(f"\t{self.gen_atom2(k, c, 'X', '_')}.")
        lines.append("")
        return lines

    def gen_exists_incompossible(self,
                                 k: PredicateType,
                                 pair: Tuple[Concept, List[Type]]
                                ) -> List[str]:
        c, ts = pair
        t1, t2 = ts
        return [
            f"% Incompossibility for {c}",
            f"incompossible({self.gen_sentence2(k, c, 'X', 'Y')}, {self.gen_sentence2(k, c, 'X', 'Y2')}) :-",
            f"\tpermanent(isa({t1}, X)),",
            f"\tpermanent(isa({t2}, Y)),",
            f"\tpermanent(isa({t2}, Y2)),",
            "\tY != Y2.",
            ""
        ]
    
    def gen_xor_constraints(self) -> List[str]:
        c_cs = []
        for t, cs in self.gen_unary_concepts().items():
            c_cs.extend(self.gen_xor_constraints_for_type(PredicateType.IS_FLUENT, t, sorted(cs)))
        c_ps = []
        for t, cs in self.gen_permanent_concepts().items():
            c_ps.extend(self.gen_xor_constraints_for_type(PredicateType.IS_PERMANENT, t, sorted(cs)))
        return c_cs + c_ps

    @staticmethod
    def gen_xor_constraints_for_type(
        k: PredicateType, t: Type, ps: List[Concept]
    ) -> List[str]:
        gs = Frame.group_predicates(ps)
        zs = list(zip(
            [Frame.group_id(k, t, i) for i in range(1, len(gs) + 1)],
            gs
        ))
        h1 = "% Choose xor group"
        cgs = Frame.choose_group(k, t, len(gs))
        h2 = "% Concept hierarchy"
        scs = [h2] + [f"sub_concept({p}, {t})." for p in ps] + [""]
        cgs_block = cgs
        r = [h1, cgs_block, ""] + scs + [
            line for (g, group) in zs for line in Frame.gen_xor_constraints_for_group(k, t, (g, group))
        ]
        return r

    @staticmethod
    def choose_group(k: PredicateType, t: Type, n: int) -> str:
        m = "; ".join(
            Frame.group_id(k, t, i) for i in range(1, n + 1)
        )
        return f"1 {{ {m} }} 1."

    @staticmethod
    def group_id(k: PredicateType, t: Type, i: int) -> str:
        t_str = str(t)[2:] if isinstance(t, str) and len(t) > 2 else str(t)
        return f"xor_{k.value}_{t_str}_{i}"

    @staticmethod
    def gen_xor_constraints_for_group(
        k: PredicateType, t: Type, gp: Tuple[str, List[List[Concept]]]
    ) -> List[str]:
        g, pss = gp
        return [
            line
            for group in pss
            for line in Frame.gen_xor_constraints_for_predicates(k, t, g, group)
        ]

    @staticmethod
    def gen_xor_constraints_for_predicates(
        k: PredicateType, t: Type, g: str, ps: List[Concept]
    ) -> List[str]:
        pairs = [(p1, p2) for p1 in ps for p2 in ps if p1 < p2]
        pss = [str(p) for p in ps]
        h1 = f"% At most one of {', '.join(pss)}"
        at_most = [h1] + [l for pair in pairs for l in Frame.gen_at_most(k, t, g, pair)]
        h2 = f"% Incompossibility {', '.join(pss)}"
        incompossibles = [h2] + [l for pair in pairs for l in Frame.gen_incompossibles(k, t, g, pair)] + [l for pair in pairs for l in Frame.gen_incompatible_unary_predicates(k, t, g, pair)]
        h3 = f"% At least one of {', '.join(pss)}"
        at_least = [h3] + Frame.gen_at_least(k, t, g, ps)
        output = Frame.gen_output(g, ps)
        return at_most + incompossibles + at_least + output

    @staticmethod
    def gen_output(g: str, ps: List[Concept]) -> List[str]:
        xor_sym = "⊕" if flag_unicode else "+"
        h = "% Readable exclusion"
        l1 = "exclusion_output(\"" + xor_sym.join(str(p) for p in ps) + "\") :-"
        l2 = f"\t{g}."
        return [h, l1, l2, ""]

    @staticmethod
    def gen_at_most(
        k: PredicateType, t: Type, g: str, pair: Tuple[Concept, Concept]
    ) -> List[str]:
        p1, p2 = pair
        return [
            ":-",
            f"\t{Frame.gen_atom(k, p1)},",
            f"\t{Frame.gen_atom(k, p2)},",
            f"\t{g}.",
            ""
        ]

    @staticmethod
    def gen_at_least(
        k: PredicateType, t: Type, g: str, ps: List[Concept]
    ) -> List[str]:
        lines = [
            ":-",
            f"\tpermanent(isa({t}, X)),",
            "\tis_time(T),",
            f"\t{g},"
        ]
        for i, p in enumerate(ps, start=1):
            atom = Frame.gen_atom(k, p)
            sep = "," if i < len(ps) else "."
            lines.append(f"\tnot {atom}{sep}")
        lines.append("")
        return lines

    @staticmethod
    def gen_incompatible_unary_predicates(
        k: PredicateType, t: Type, g: str, pair: Tuple[Concept, Concept]
    ) -> List[str]:
        p1, p2 = pair
        return [
            f"incompatible_unary_predicates({p1}, {p2}) :-",
            f"\t{g}.",
            ""
        ]

    @staticmethod
    def gen_incompossibles(
        k: PredicateType, t: Type, g: str, pair: Tuple[Concept, Concept]
    ) -> List[str]:
        p1, p2 = pair
        return [
            f"incompossible({Frame.gen_sentence(k, p1)}, {Frame.gen_sentence(k, p2)}) :-",
            f"\tpermanent(isa({t}, X)),",
            f"\t{g}.",
            ""
        ]

    @staticmethod
    def gen_atom(k: PredicateType, p: Concept) -> str:
        if k == PredicateType.IS_FLUENT:
            return f"holds(s({p}, X), T)"
        return f"permanent(isa({p}, X))"

    @staticmethod
    def gen_atom2(
        k: PredicateType, p: Concept, v1: str, v2: str
    ) -> str:
        if k == PredicateType.IS_FLUENT:
            return f"holds(s2({p}, {v1}, {v2}), T)"
        return f"permanent(isa2({p}, {v1}, {v2}))"

    @staticmethod
    def gen_sentence(k: PredicateType, p: Concept) -> str:
        if k == PredicateType.IS_FLUENT:
            return f"s({p}, X)"
        return f"isa({p}, X)"

    @staticmethod
    def gen_sentence2(
        k: PredicateType, p: Concept, v1: str, v2: str
    ) -> str:
        if k == PredicateType.IS_FLUENT:
            return f"s2({p}, {v1}, {v2})"
        return f"isa2({p}, {v1}, {v2})"

    @staticmethod
    def choose_group(k: PredicateType, t: Type, n: int) -> str:
        opts = "; ".join(
            f"{Frame.group_id(k, t, i)}"
            for i in range(1, n+1)
        )
        return f"1 {{ {opts} }} 1."

    @staticmethod
    def group_id(k: PredicateType, t: Type, i: int) -> str:
        type_name = t.name[2:] if t.name.startswith("T_") else t.name
        return f"xor_{k.name.lower()}_{type_name}_{i}"

    @staticmethod
    def xor_group(k: PredicateType, t: Type, ps: List[Concept]) -> List[str]:
        return [Frame.group_id(k, t, i+1) for i in range(len(Frame.group_predicates(ps)))]

    @staticmethod
    def group_predicates(xs: List) -> List[List[List]]:
        """
        Return all partitions of xs into contiguous sublists
        where each sublist has length >= 2.
        """
        return [
            grouping
            for grouping in Frame.divide_into_groups(xs)
            if all(len(group) >= 2 for group in grouping)
        ]

    @staticmethod
    def divide_into_groups(xs: List) -> List[List[List]]:
        """
        Generate every contiguous grouping of xs, without duplicates.
        """
        raw = Frame.divide_into_groups2(xs, [[]])
        seen = set()
        result: List[List[List]] = []
        for grouping in raw:
            # reverse each sublist and the outer list
            rev_sublists = [list(reversed(gr)) for gr in grouping]
            final = list(reversed(rev_sublists))
            key: Tuple[Tuple, ...] = tuple(tuple(gr) for gr in final)
            if key not in seen:
                seen.add(key)
                result.append(final)
        return result

    @staticmethod
    def divide_into_groups2(
        xs: List, acc: List[List[List]]
    ) -> List[List[List]]:
        """
        Recursive helper: at each element, either insert into the reversed 'last' sublist
        or start a new one, building all combinations.
        """
        if not xs:
            return acc

        x, *rest = xs
        acc2 = Frame.insert_in_last_list(x, acc)
        acc3 = Frame.make_new_list(x, acc)
        return (
            Frame.divide_into_groups2(rest, acc2)
            + Frame.divide_into_groups2(rest, acc3)
        )

    @staticmethod
    def insert_in_last_list(
        x: Any, gs: List[List[List]]
    ) -> List[List[List]]:
        """
        (Reversed semantics) insert x into the head-sublist of each grouping.
        """
        out: List[List[List]] = []
        for grouping in gs:
            if not grouping:
                out.append([[x]])
            else:
                head, *tail = grouping
                out.append([[x] + head] + tail)
        return out

    @staticmethod
    def make_new_list(
        x: Any, gs: List[List[List]]
    ) -> List[List[List]]:
        """
        (Reversed semantics) start a new sublist [x] at the head of each grouping.
        """
        out: List[List[List]] = []
        for grouping in gs:
            if not grouping:
                out.append([[x]])
            else:
                out.append([[x]] + grouping)
        return out

@dataclass
class Template:
    dir: str
    frame: Frame = field(compare=False)
    min_body_atoms: int
    max_body_atoms: int
    num_arrow_rules: int
    num_causes_rules: int
    use_noise: bool
    
    num_visual_predicates: Optional[int] = None

    def gen_interpretation(self, name: str) -> None:
        """Write the interpretation file in one pass."""
        path = f"temp/{name}_interpretation.lp"
        lines = [
            "% Auto-generated from GenInterpretation",
            "#program base.",
            *self.interpretation_lines(),
        ]
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"Generated {path}")

    def interpretation_lines(self) -> List[str]:
        es = self.frame.gen_elements()
        ts = self.gen_typing()
        crs, n = self.gen_conceptual_rules()
        urs = self.gen_update_rules(n)
        cs = self.gen_constraints()
        vs = VisualCodeGenerator.gen_visual_code(self) if self.num_visual_predicates is not None else []
        stats = [
            divider, "% Stats", divider, "",
            f"num_objects({len(self.get_objects())}).",
            f"num_variables({len(self.frame.vars)})."
        ]
        return es + ts + crs + urs + cs + vs + stats

    def gen_typing(self) -> List[str]:
        objs = self.get_objects()
        xs = [f"permanent(isa({ty}, {obj}))." for obj, ty in objs]
        sts = [f"sub_type({sub}, {sup})." for sub, sup in self.frame.sub_types()]
        return [divider, "% Typing", divider, ""] + xs + [""] + sts + [""]

    def gen_update_rules(self, n: int) -> List[str]:
        hdr = [divider, "% Update rules", divider, ""]
        cs = [
            "1 { rule_var_group(R, VG) : is_var_group(VG) } 1 :- is_gen_rule(R), use_rule(R).", "",
            f"{self.min_body_atoms} {{ rule_body(R, VA) : is_var_atom(VA) }} "
            f"{self.max_body_atoms} :- is_gen_rule(R), use_rule(R).", "",
            "1 { rule_causes_head(R, VA) : cause_head(VA) } 1 :- is_causes_rule(R), use_rule(R).", "",
            "1 { rule_arrow_head(R, VA) : is_var_fluent(VA) } 1 :- is_arrow_rule(R), use_rule(R).", ""
        ]
        arrows = [f"is_arrow_rule(r{i})." for i in range(n, n + self.num_arrow_rules)]
        n2 = n + self.num_arrow_rules
        causes = [f"is_causes_rule(r{i})." for i in range(n2, n2 + self.num_causes_rules)]
        uses = [
            "{ use_rule(R) } :- is_arrow_rule(R).",
            "{ use_rule(R) } :- is_causes_rule(R)."
        ]
        return hdr + cs + arrows + causes + uses + [""]

    def gen_constraints(self) -> List[str]:
        if True :
            blind = ["% [Ignoring Kantian blind sense condition]"]
        else:
            blind = (
                [":- violation_kant_condition_blind_sense."]
                if not self.use_noise
                else ["% Adding noise", ":~ senses(S, T), not holds(S, T). [1 @ 1, S, T]"]
            )

        if True:
            spatial = ["% [Ignoring Kantian spatial unity condition]", ""]
        else:
            spatial = [":- violation_kant_condition_spatial_unity.", ""]

        noise_flag = ["% Adding noise", "flag_is_using_noise."] if self.use_noise else []
        return [divider, "% Constraints", divider, ""] + blind + spatial + noise_flag

    def gen_inits(self, name: str) -> None:
        """Write all initial atoms in one go."""
        path = f"temp/{name}_init.lp"
        atoms = [a for a in self.all_ground_atoms() if not self.is_static_atom(a)]
        lines = [
            "% Auto-generated from GenInterpretation",
            "#program base.",
            *[f"{{ {a} }} ." for a in atoms],
        ]
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"Generated {path}")

    def is_static_atom(self, a: GroundAtom) -> bool:
        return isinstance(a, (GA, Perm)) and a.concept in self.frame.static_concepts

    def gen_subs(self, name: str) -> None:
        """Generate variable groups and substitutions in one pass."""
        path = f"temp/{name}_subs.lp"
        frm = self.frame

        lines = [
            "% Auto-generated from GenInterpretation",
            "#program base. ",
            "%-----------",
            "% var_types",
            "%-----------",
            "",
            *[f"var_type({v}, {ty})." for v, ty in frm.vars],
            "",
            "%--------------",
            "% contains_var",
            "%--------------",
            "",
        ]

        for vg in frm.var_groups:
            n = self.frame.var_group_name(vg)
            for v in vg:
                lines.append(f"contains_var(var_group_{n}, {v}).")
            lines.append("")

        lines.extend([
            "",
            "%----------",
            "% subs",
            "%----------",
            "",
        ])

        for group_name, groups in self.all_subs():
            for i, subs in enumerate(groups, start=1):
                lines.append(
                    f"subs_group(var_group_{group_name}, subs_{group_name}_{i})."
                )
                for v, x in subs:
                    lines.append(f"subs(subs_{group_name}_{i}, {v}, {x}).")
                lines.append("")
            lines.append("")

        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"Generated {path}")

    def print_subs_group(self, file: str, sg: Tuple[str, List[Any]]) -> None:
        name, ss = sg
        for i, subs in enumerate(ss, start=1):
            self.print_subs(file, name, (i, subs))
        with open(file, "a") as fh:
            fh.write("\n")

    def print_subs(self, file: str, name: str, pair: Tuple[int, List[Tuple[Any, Any]]]) -> None:
        i, subs = pair
        with open(file, "a") as fh:
            fh.write(f"subs_group(var_group_{name}, subs_{name}_{i}).\n")
            for v, x in subs:
                fh.write(f"subs(subs_{name}_{i}, {v}, {x}).\n")
            fh.write("\n")

    def gen_var_atoms(self, name: str) -> None:
        """Write all variable atoms to disk with a single file write."""
        path = f"temp/{name}_var_atoms.lp"
        frm = self.frame
        lines = [
            "% Auto-generated from GenInterpretation",
            "#program base.",
        ]

        for a in frm.all_var_fluents():
            for vg in frm.all_var_groups(a):
                lines.append(
                    f"var_fluent({a}, var_group_{frm.var_group_name(vg)})."
                )

        for a in frm.all_var_isas():
            for vg in frm.all_var_groups(a):
                lines.append(
                    f"var_permanent({a}, var_group_{frm.var_group_name(vg)})."
                )

        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"Generated {path}")

    # ——— simple forwarders ———
    def get_fluid_concepts(self):     return self.frame.fluid_concepts
    def get_permanent_concepts(self): return self.frame.permanent_concepts
    def get_input_concepts(self):     return self.frame.input_concepts
    def get_objects(self):            return self.frame.objects
    def get_exogeneous_objects(self): return self.frame.exogeneous_objects

    def all_ground_atoms(self) -> List[GroundAtom]:
        ss: List[GroundAtom] = []
        for c, ts in self.get_fluid_concepts():
            for xs in self.all_obj_tuples(ts):
                ss.append(GA(c, xs))

        sts = self.frame.sub_types_star()
        for p, lineage, types in self.get_permanent_concepts():
            if lineage is ConceptLineage.CONSTRUCTED:
                if len(types) == 1:
                    t, = types
                    for x, tp in self.get_objects():
                        if (tp, t) in sts:
                            ss.append(Perm(p, [x]))
                elif len(types) == 2:
                    t1, t2 = types
                    for x, tp1 in self.get_objects():
                        if (tp1, t1) in sts:
                            for y, tp2 in self.get_objects():
                                if (tp2, t2) in sts:
                                    ss.append(Perm(p, [x, y]))
        return ss

    def all_obj_tuples(self, types_list: List[Any]) -> List[List[Any]]:
        if not types_list:
            return [[]]
        first, *rest = types_list
        sts = self.frame.sub_types_star()
        objs = [o for o, tp in self.get_objects() if (tp, first) in sts]
        result: List[List[Any]] = []
        for o in objs:
            for tail in self.all_obj_tuples(rest):
                result.append([o] + tail)
        return result

    def all_subs(self) -> List[Tuple[str, List[List[Tuple[Any, Any]]]]]:
        subs_groups = []
        sub_types_list = Template.fixed_point(Template.trans, self.frame.sub_types())
        for vg in self.frame.var_groups:
            name = self.frame.var_group_name(vg)
            typed_vs = self.make_var_group(self.frame.vars, vg)
            groups = Template.gen2(self.get_objects(), sub_types_list, typed_vs)
            subs_groups.append((name, groups))
        return subs_groups

    def make_var_group(self,
                       types_list: List[Tuple[Any, Any]],
                       vs: List[Any]
                      ) -> List[Tuple[Any, Any]]:
        result = []
        for v in vs:
            for tv, tt in types_list:
                if tv == v:
                    result.append((v, tt))
                    break
            else:
                raise ValueError(f"No type for var {v!r}")
        return result


    def gen_conceptual_rules(self) -> Tuple[List[str], int]:
        if flag_ablation_remove_kant_condition_conceptual_unity:
            return (["", "% [Ignoring conceptual unity condition]", ""], 1)
        h = [divider, "% Conceptual structure", divider, ""]
        return (h + self.frame.gen_xor_constraints() + self.frame.gen_exists_constraints() , 1)

    @staticmethod
    def fixed_point(f, a):
        while True:
            na = f(a)
            if na == a:
                return a
            a = na

    @staticmethod
    def trans(xs: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
        rs = set(xs)
        for x, z in xs:
            for z2, y in xs:
                if z == z2:
                    rs.add((x, y))
        return sorted(rs)

    @staticmethod
    def gen2(objects: List[Tuple[Any, Any]],
             sub_types_list: List[Tuple[Any, Any]],
             vars_typed: List[Tuple[Any, Any]]
            ) -> List[List[Tuple[Any, Any]]]:
        if not vars_typed:
            return [[]]
        v, t = vars_typed[0]
        rest = vars_typed[1:]
        candidates = [(v, o) for o, tp in objects if Template.is_sub_type(sub_types_list, tp, t)]
        result = []
        for pick in candidates:
            for tail in Template.gen2(objects, sub_types_list, rest):
                result.append([pick] + tail)
        return result

    @staticmethod
    def is_sub_type(sub_types_list: List[Tuple[Any, Any]], x: Any, y: Any) -> bool:
        return x == y or (x, y) in sub_types_list

class VisualCodeGenerator:
    @staticmethod
    def all_lists_from(values: List[Any], length: int) -> List[List[Any]]:
        if length == 0:
            return [[]]
        return [ [v] + rest
                 for v in values
                 for rest in VisualCodeGenerator.all_lists_from(values, length - 1) ]

    @staticmethod
    def all_unary_predicate_assignments(num_preds: int, num_objs: int) -> List[List[int]]:
        if num_preds > num_objs:
            raise ValueError("num_preds exceeds num_objs")
        domain = list(range(1, num_preds + 1))
        candidates = VisualCodeGenerator.all_lists_from(domain, num_objs)
        def cond1(lst): return all(p in lst for p in domain)
        def cond2(lst): return all(lst[i] <= lst[j]
                                    for i in range(len(lst)) for j in range(i, len(lst)))
        def cond3(lst): return all(lst.count(i) <= lst.count(i + 1)
                                    for i in range(1, num_preds))
        return [lst for lst in candidates if cond1(lst) and cond2(lst) and cond3(lst)]

    @staticmethod
    def gen_visual_mappings(objs: List[str], num_preds: int) -> List[str]:
        ms = VisualCodeGenerator.all_unary_predicate_assignments(num_preds, len(objs))
        lines: List[str] = [
            "1 { visual_mapping(M) : in_visual_mapping_range(M) } 1.",
            "",
            f"in_visual_mapping_range(1..{len(ms)})."
        ]
        for i, m in enumerate(ms, start=1):
            for obj, p in zip(objs, m):
                lines.append(f"is_visual_type({obj}, vt_type_{p}) :- visual_mapping({i}).")
        return lines

    @staticmethod
    def sprite_type_text(num_preds: int) -> List[str]:
        lines: List[str] = [""]
        ts = [f"type_{p}" for p in range(num_preds + 1)]
        for i, t in enumerate(ts, start=1):
            lines.append(f"sprite_type(E, vt_{t}) :-")
            for j, _ in enumerate(ts, start=1):
                sep = "," if j < len(ts) else "."
                val = 1 if i == j else 0
                lines.append(f"\tbnn_result(E, {j}, {val}){sep}")
            lines.append("")
        return lines

    @staticmethod
    def looks_constraints_text(num_preds: int) -> List[str]:
        lines: List[str] = []
        ts = [f"type_{p}" for p in range(num_preds + 1)]
        for t in ts:
            lines.append(f":- not some_looks_type(vt_{t}).")
            lines.append(f"some_looks_type(vt_{t}) :- contains_visual_type(C, vt_{t}, T).")
        return lines

    @staticmethod
    def bnn_text(num_preds: int) -> List[str]:
        return ["", f"nodes(1, 25).", f"nodes(2, 10).", f"nodes(3, {num_preds + 1})."]

    @staticmethod
    def num_object_types(tpl: Any) -> int:
        # assume "cell" is one of the types -> subtract 1
        return len(tpl.frame.types) - 1

    @staticmethod
    def senses_clauses(n: int) -> List[str]:
        lines: List[str] = []
        for i in range(1, n + 1):
            lines.extend([
                f"1 {{ senses(s2(c_in_{i}, Obj, Cell), T) : contains_visual_type(Cell, VT, T) }} 1 :-",
                f"\tis_visual_type(Obj, VT),",
                f"\tVT = vt_type_{i},",
                f"\tis_time(T),",
                f"\tnot is_test_time(T).",
                ""
            ])
        return lines

    @staticmethod
    def some_type_at_clauses(n: int) -> List[str]:
        lines: List[str] = []
        for i in range(1, n + 1):
            lines.extend([
                "some_type_at(Cell, VT, T) :- ",
                f"\tsenses(s2(c_in_{i}, X, Cell), T),",
                "\tis_visual_type(X, VT).",
                ""
            ])
        return lines

    @staticmethod
    def something_at_clauses(n: int) -> List[str]:
        lines: List[str] = []
        for i in range(1, n + 1):
            lines.extend([
                "something_at(T, C) :- ",
                "\tis_test_time(T),",
                f"\tholds(s2(c_in_{i}, Obj, C), T).",
                ""
            ])
        return lines

    @staticmethod
    def possible_sprite_at_clauses(n: int) -> List[str]:
        lines: List[str] = []
        for i in range(1, n + 1):
            lines.extend([
                "possible_sprite_at(T, C, S) :- ",
                "\tis_test_time(T), ",
                f"\tholds(s2(c_in_{i}, Obj, C), T), ",
                "\tis_visual_type(Obj, VT), ",
                "\tsprite_type(S, VT).    ",
                ""
            ])
        return lines

    @staticmethod
    def visual_sokoban_clauses(tpl: Any) -> List[str]:
        n = VisualCodeGenerator.num_object_types(tpl)
        return (
            VisualCodeGenerator.senses_clauses(n)
            + [""]
            + VisualCodeGenerator.some_type_at_clauses(n)
            + [""]
            + VisualCodeGenerator.something_at_clauses(n)
            + [""]
            + VisualCodeGenerator.possible_sprite_at_clauses(n)
            + [""]
        )

    @staticmethod
    def gen_visual_code(tpl: Template) -> List[str]:
        divider = "% -----------------------------"
        h = [divider, "% Low-level visual processing", divider, ""]
        num_preds = tpl.num_visual_predicates
        objs = [str(obj) for obj, _ in tpl.frame.objects
                if hasattr(obj, 'name') and not obj.name.startswith('c')]
        return (
            h
            + VisualCodeGenerator.gen_visual_mappings(objs, num_preds)
            + VisualCodeGenerator.sprite_type_text(num_preds)
            + VisualCodeGenerator.looks_constraints_text(num_preds)
            + VisualCodeGenerator.bnn_text(num_preds)
            + [""]
            + VisualCodeGenerator.visual_sokoban_clauses(tpl)
        )
