import itertools
from dataclasses import dataclass
from typing import List, Tuple, Union, Iterator, Iterable
from Interpretation import Frame, Template, Type as T, Object as O, Var as V, Concept, Object, Var, ConceptLineage
from itertools import count, product


# ----------------------- Template-iteration utilities -----------------------

ConceptSpec = Tuple[Concept, List[T]]
ObjectSpec = Tuple[Object, T]
VarSpec = Tuple[Var, T]

@dataclass(frozen=True)
class IP_ObjectSpecs:
    specs: List[ObjectSpec]

@dataclass(frozen=True)
class IP_PermConcept:
    specs: List[ConceptSpec]

@dataclass(frozen=True)
class IP_FluentConcept:
    specs: List[ConceptSpec]

@dataclass(frozen=True)
class IP_VarSpecs:
    specs: List[VarSpec]

@dataclass(frozen=True)
class IP_NumArrowRules:
    count: int

@dataclass(frozen=True)
class IP_NumCausesRules:
    count: int

@dataclass(frozen=True)
class IP_NumBodyAtoms:
    count: int

TemplateParameter = Union[IP_ObjectSpecs, IP_PermConcept, IP_FluentConcept,
                          IP_VarSpecs, IP_NumArrowRules, IP_NumCausesRules,
                          IP_NumBodyAtoms]

@dataclass
class TemplateDelta:
    extra_types: List[T]
    extra_objects: List[ObjectSpec]
    extra_perm_concepts: List[ConceptSpec]
    extra_fluent_concepts: List[ConceptSpec]
    extra_vars: List[VarSpec]
    extra_num_arrow_rules: int
    extra_num_causes_rules: int
    extra_num_body_atoms: int

IntTypesPair = Tuple[int, List[T]]
const_num_templates_per_type = 100


def get_first_n_int_pairs(n: int) -> List[Tuple[int, int]]:
    """
    Generate the first n "units" of (count, types) pairs as in Haskell implementation.
    """
    result = []
    total = 0
    for i, y in zip(count(1), count(0)):
        block = const_num_templates_per_type * i
        if total + block >= n:
            diff = n - total
            if diff > 0:
                result.append((diff, y))
            break
        result.append((block, y))
        total += block
    return result


def get_first_n_int_types_pairs(n: int) -> List[IntTypesPair]:
    return [(num, [T(f"gen_{i}") for i in range(1, cnt+1)]) 
            for num, cnt in get_first_n_int_pairs(n)]


def choices(lists: List[List[TemplateParameter]]) -> Iterator[List[TemplateParameter]]:
    """Cartesian product over lists of TemplateParameter, similar to Data.Universe.Helpers.choices"""
    return product(*lists)


def parameter_lists(types: List[T], n: int) -> List[TemplateDelta]:
    deltas: List[TemplateDelta] = []
    for count, new_types in get_first_n_int_types_pairs(n):
        deltas.extend(parameter_lists2(types + new_types, new_types, count))
    return deltas


def parameter_lists2(all_types: List[T], new_types: List[T], n: int) -> List[TemplateDelta]:
    return list(all_parameter_lists(all_types, new_types))[:n]


def all_parameter_lists(all_ts: List[T], new_ts: List[T]) -> Iterator[TemplateDelta]:
    num_body_atoms = [IP_NumBodyAtoms(i) for i in count(0)]
    fluents = [IP_FluentConcept(cs) for cs in all_concepts(all_ts)]
    object_specs = [IP_ObjectSpecs(objs) for objs in all_object_specs(all_ts)]
    perms = [IP_PermConcept(cs) for cs in all_concepts(all_ts)]
    num_arrows = [IP_NumArrowRules(i) for i in count(0)]
    num_causes = [IP_NumCausesRules(i) for i in count(0)]
    vars_ = [IP_VarSpecs(vs) for vs in all_var_specs(all_ts)]
    for combo in choices([num_body_atoms, fluents, object_specs, perms,
                          num_arrows, num_causes, vars_]):
        yield convert_to_td(new_ts, list(combo))


def convert_to_td(new_ts: List[T], params: List[TemplateParameter]) -> TemplateDelta:
    # Unpack parameters by type
    n_body = next(p.count for p in params if isinstance(p, IP_NumBodyAtoms))
    fluent = next(p.specs for p in params if isinstance(p, IP_FluentConcept))
    objs = next(p.specs for p in params if isinstance(p, IP_ObjectSpecs))
    perm = next(p.specs for p in params if isinstance(p, IP_PermConcept))
    n_arw = next(p.count for p in params if isinstance(p, IP_NumArrowRules))
    n_cau = next(p.count for p in params if isinstance(p, IP_NumCausesRules))
    vars_ = next(p.specs for p in params if isinstance(p, IP_VarSpecs))
    return TemplateDelta(
        extra_types=new_ts,
        extra_objects=objs,
        extra_perm_concepts=perm,
        extra_fluent_concepts=fluent,
        extra_vars=vars_,
        extra_num_arrow_rules=n_arw,
        extra_num_causes_rules=n_cau,
        extra_num_body_atoms=n_body
    )


def augment_template(template: Template, td: TemplateDelta) -> Template:
    # Create a copy and augment
    frame = template.frame.copy()
    frame.types += td.extra_types
    frame.objects += td.extra_objects
    frame.permanent_concepts += [(c, 'Constructed', ts) for c, ts in td.extra_perm_concepts]
    frame.fluid_concepts += td.extra_fluent_concepts
    frame.vars += td.extra_vars
    if td.extra_vars:
        frame.var_groups.append(frame.var_groups[-1] + [v for v, _ in td.extra_vars])

    return Template(
        frame=frame,
        num_arrow_rules=template.num_arrow_rules + td.extra_num_arrow_rules,
        num_causes_rules=template.num_causes_rules + td.extra_num_causes_rules,
        max_body_atoms=template.max_body_atoms + td.extra_num_body_atoms,
        **{k: getattr(template, k) for k in template.__dict__ 
           if k not in ('frame', 'num_arrow_rules', 'num_causes_rules', 'max_body_atoms')}
    )


def show_parameters(td: TemplateDelta) -> str:
    parts: List[str] = []
    parts += (["No extra types"] if not td.extra_types else ["Extra types:"] + [str(t) for t in td.extra_types])
    parts += (["No extra objects"] if not td.extra_objects else ["Extra objects:"] + [str(o) for o in td.extra_objects])
    parts += (["No extra permanent concepts"] if not td.extra_perm_concepts else ["Extra permanent concepts:"] + [str(c) for c in td.extra_perm_concepts])
    parts += (["No extra fluent concepts"] if not td.extra_fluent_concepts else ["Extra fluent concepts:"] + [str(f) for f in td.extra_fluent_concepts])
    parts += (["No extra vars"] if not td.extra_vars else ["Extra vars:"] + [str(v) for v in td.extra_vars])
    parts.append(f"Num extra arrow rules: {td.extra_num_arrow_rules}")
    parts.append(f"Num extra causes rules: {td.extra_num_causes_rules}")
    parts.append(f"Num extra body atoms: {td.extra_num_body_atoms}")
    return "\n".join(parts)


def all_concepts(types: List[T]) -> Iterator[List[ConceptSpec]]:
    length = 0
    while True:
        for combo in all_concepts_of_length(types, length):
            yield combo
        length += 1


def all_concepts_of_length(types: List[T], n: int) -> List[List[ConceptSpec]]:
    if n == 0:
        return [[]]
    combos: List[List[ConceptSpec]] = []
    for prev in all_concepts_of_length(types, n-1):
        for spec in all_concepts_with_index(types, n):
            combos.append(prev + [spec])
    return combos


def all_concepts_with_index(types: List[T], i: int) -> List[ConceptSpec]:
    unaries = [(Concept(f"gen_{i}"), [t]) for t in types]
    binaries = [(Concept(f"gen_{i}"), [t1, t2]) for t1 in types for t2 in types]
    return unaries + binaries


def all_object_specs(types: List[T]) -> Iterator[List[ObjectSpec]]:
    length = 0
    while True:
        for combo in all_object_specs_of_length(types, length):
            yield combo
        length += 1


def all_object_specs_of_length(types: List[T], n: int) -> List[List[ObjectSpec]]:
    if n == 0:
        return [[]]
    combos: List[List[ObjectSpec]] = []
    for prev in all_object_specs_of_length(types, n-1):
        for spec in all_object_specs_with_index(types, n):
            combos.append(prev + [spec])
    return combos


def all_object_specs_with_index(types: List[T], i: int) -> List[ObjectSpec]:
    return [(Object(f"gen_{i}"), t) for t in types]


def all_var_specs(types: List[T]) -> Iterator[List[VarSpec]]:
    length = 0
    while True:
        for combo in all_var_specs_of_length(types, length):
            yield combo
        length += 1


def all_var_specs_of_length(types: List[T], n: int) -> List[List[VarSpec]]:
    if n == 0:
        return [[]]
    combos: List[List[VarSpec]] = []
    for prev in all_var_specs_of_length(types, n-1):
        for spec in all_var_specs_with_index(types, n):
            combos.append(prev + [spec])
    return combos


def all_var_specs_with_index(types: List[T], i: int) -> List[VarSpec]:
    return [(Var(f"gen_{i}"), t) for t in types]