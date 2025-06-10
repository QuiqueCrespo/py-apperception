from dataclasses import dataclass, field
from typing import List, Any, Union, Dict, Tuple
from Interpretation import Template, InterpretationStatistics, Interpretation, Frame, Rule, Arrow
import re
import os


# ----------------------------------------
# Data structures for Clingo outputs
# ----------------------------------------
@dataclass(frozen=True)
class ClingoOutput:
    """Base class for Clingo outputs."""

@dataclass(frozen=True)
class Answer(ClingoOutput):
    answer: str

@dataclass(frozen=True)
class Optimization(ClingoOutput):
    optimization: str

# Union type for outputs
ClingoOutputType = Union[Answer, Optimization]

@dataclass
class ClingoResult:
    result_answer: str
    result_optimization: str
    result_template: Template

# ----------------------------------------
# Parser for Clingo textual output
# ----------------------------------------
class ClingoParser:
    OPT_PREFIX = "Optimization: "

    def __init__(self, show_answer_set: bool = False, show_extraction: bool = False):
        self.show_answer_set = show_answer_set
        self.show_extraction = show_extraction

    def parse_lines(self, lines: List[str]) -> List[ClingoOutputType]:
        """
        Parse raw Clingo output lines into Answer/Optimization objects.
        """
        outputs: List[ClingoOutputType] = []
        i = 0
        while i < len(lines):

            outputs.append(Answer(answer=lines[i ]))
                
            i += 1
        return outputs

    def last_outputs(self, outputs: List[Answer]) -> List[ClingoOutputType]:
        """
        Return only the final answer or the answer+optimum pair.
        """
        if not outputs:
            return []
        last = outputs[-1]

    
        if isinstance(last, Answer):
            return [last]
        if isinstance(last, Optimization) and len(outputs) >= 2:
            return [outputs[-2], last]
        return []

# ----------------------------------------
# Presenter: formatting of answers
# ----------------------------------------
class ClingoPresenter:
    def __init__(self, show_answer_set: bool = False, show_extraction: bool = False, flag_output_latex: bool = False):
        self.show_answer_set = show_answer_set
        self.show_extraction = show_extraction
        self.flag_output_latex = flag_output_latex

    def present(self, template: Any, output: ClingoOutputType) -> str:
        """Produce a formatted string for a ClingoOutput."""
        if isinstance(output, Answer):
            return self._present_answer(template, output)
        if isinstance(output, Optimization):
            return f"{ClingoParser.OPT_PREFIX}{output.optimization}"
        return ""

    def _present_answer(self, template: Any, output: Answer) -> str:
        lines: List[str] = ["-------------", "Answer", "-------------", ""]
        words = output.answer.split()
        if self.show_answer_set:
            lines.extend(sorted(words))
            lines.append("")
        filtered = [w for w in words if "wibble" not in w]
        if self.show_extraction:
            interp = Interpretation(filtered)
            if self.flag_output_latex:
                lines.extend(self.readable_interpretation(interp) + latex_output(template, interp))
            else:
                lines.extend(self.readable_interpretation(interp))
        return "\n".join(lines)

# ----------------------------------------
# Interpretation display utilities
# ----------------------------------------

    def readable_interpretation(self, interp: 'Interpretation') -> List[str]:
        lines: List[str] = []
        if interp.inits:
            lines.extend(["", "Initial conditions", "------------------", ""] + interp.inits)
        if interp.permanents:
            lines.extend(["", "Permanents", "----------", ""] + interp.permanents)
        lines.extend(["", "Rules", "-----", ""] + [str(r) for r in interp.rules])
        if interp.exclusions:
            lines.extend(["", "Constraints", "-----------", ""] + interp.exclusions)
        lines.extend(["", "Trace", "-----", ""] + [self.show_fact(t, fs) for t, fs in interp.facts])
        lines.extend(["", "Accuracy", "--------", ""])
        lines.append("Status: correct" if interp.correct else "Status: incorrect")
        if interp.num_accurate is not None and interp.num_held_outs is not None:
            p = (interp.num_accurate / interp.num_held_outs
                if interp.num_held_outs else float('nan'))
            lines.extend(["", f"Percentage accurate: {p}"])
        lines.extend(self.readable_stats(interp.statistics))
        return lines


    def show_fact(self, time: int, facts: List[str]) -> str:
        return f"Time {time}: \n" + "\n ".join(facts)


    def readable_stats(self, stats: 'InterpretationStatistics') -> List[str]:
        out = [
            "", "Statistics", "----------", "",
            f"Num arrow rules: {stats.num_used_arrow_rules}",
            f"Num causes rules: {stats.num_used_causes_rules}",
            f"Total body atoms: {stats.total_body_atoms}",
            f"Num inits: {stats.num_inits}",
            f"Total cost: {stats.total_cost()}",
            f"Total num clauses: {stats.total_num_clauses()}"
        ]
        if stats.bnn_entropy is not None:
            out.append(f"Entropy of bnn : {stats.bnn_entropy}")
        if stats.ambiguity is not None:
            out.extend([f"Ambiguity: {stats.ambiguity}", ""])
        if stats.possible_preds:
            out.extend(["BNN", "----", ""] + [self.show_pp(p) for p in stats.possible_preds])
        return out
    

    @staticmethod
    def show_pp( p: str) -> str:
        return f"possible_pred({p})."



def write_latex(template: Template, output: ClingoOutput, file_path: str = 'temp/results.tex') -> None:
    """
    Generate a complete LaTeX document from the given template and Clingo output.
    """

    # Only handle plain answers, skip optimizations
    if not isinstance(output, Answer):
        return

    # Parse and clean the answer
    terms = sorted(output.answer.split()) + ['']
    terms = [t for t in terms if 'wibble' not in t]
    interp = Interpretation(terms)

    # Build the document
    lines = latex_output(template, interp)

    # Write to file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Created LaTeX file at {file_path}")

def latex_output(template: Template, interp: Interpretation) -> List[str]:
    """
    Generate LaTeX output for the given template and interpretation.
    """
    lines = []
    lines += latex_given(interp)
    lines += latex_frame(template)
    lines += latex_interpretation(template, interp)
    return lines

# === LaTeX Document Skeleton ===

def latex_document_header() -> List[str]:
    return [
        r"\documentclass[12pt]{article}",
        r"\usepackage{amsmath,amssymb,booktabs,float}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        ""
    ]

def latex_document_footer() -> List[str]:
    return [
        r"\end{document}"
    ]

def latex_example(template: Template, interp: Interpretation) -> List[str]:
    return [
        r"\begin{example}",
        ""
    ] + \
    latex_given(interp) + [""] + \
    latex_frame(template) + [""] + \
    latex_interpretation(template, interp) + [""] + \
    [r"\end{example}"]


# === “Given” Section with Input Table ===

@dataclass(order=True)
class InputTableElem:
    time: int
    object: str
    attribute: str
    hidden: bool = False

@dataclass
class InputTable:
    max_time: int
    all_objects: List[str]
    attributes: Dict[Tuple[int, str], str] = field(default_factory=dict)

def input_table(times: List[int], elems: List[InputTableElem]) -> InputTable:
    max_t = max(times) if times else 0
    objs = sorted({e.object for e in elems})
    attrs: Dict[Tuple[int, str], str] = {
        (e.time, e.object): e.attribute
        for e in elems if not e.hidden
    }
    return InputTable(max_time=max_t, all_objects=objs, attributes=attrs)

def latex_given(interp: Interpretation) -> List[str]:
    header = [r"\noindent\textbf{Given the following input:}"]
    # build table elements
    ss = [(False, extract_time_pair(replace_atom(s))) for s in interp.senses]
    hs = [(True,  extract_time_pair(replace_atom(s))) for s in interp.hiddens]
    elems = [extract_input_table_elem(h, p) for h,p in (ss+hs)]
    table = input_table(interp.times, elems)
    return header + show_input_table(table)

def show_input_table(table: InputTable) -> List[str]:
    # Use booktabs for nicer tables
    cols = len(table.all_objects)
    fmt = 'l' + 'c' * cols
    header = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\begin{{tabular}}{{{fmt}}}",
        r"\toprule",
        "Time & " + " & ".join(all_objects_text(table)) + r" \\",
        r"\midrule"
    ]
    body = []
    for t in range(1, table.max_time + 1):
        row = f"{t} & " + " & ".join(
            _lookup_cell(t, o, table.attributes) for o in table.all_objects
        ) + r" \\"
        body.append(row)
    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ]
    return header + body + footer

def _lookup_cell(time: int, obj: str, attrs: Dict[Tuple[int, str], str]) -> str:
    val = attrs.get((time, obj))
    return convert_token(val) if val else r"\text{--}"

def all_objects_text(table: InputTable) -> List[str]:
    # strip any "obj_" from the column headings
    return [f"${convert_to_mathit(replace_atom(o))}$" for o in table.all_objects]

def convert_token(x: str) -> str:
    # strip prefixes from the cell contents too
    clean = replace_atom(x)
    return f"${convert_to_mathit(clean)}$"


# === Utilities to parse “time-atom” pairs ===

def extract_time_pair(s: str) -> Tuple[int, str]:
    if ',' not in s:
        raise ValueError(f"No comma in time-pair: {s}")
    obj, tm = s.rsplit(',', 1)
    return int(tm), obj

def extract_input_table_elem(hidden: bool, pair: Tuple[int, str]) -> InputTableElem:
    t, atom = pair
    obj, attr = extract_object_attribute(atom)
    return InputTableElem(time=t, object=obj, attribute=attr, hidden=hidden)

def extract_object_attribute(s: str) -> Tuple[str, str]:
    parts = split_one_of("(),", s)
    # e.g. ["s", "attr", "obj", "time"]
    if parts and parts[0] in ('s','s2') and len(parts) >= 4:
        if parts[0] == 's2':
            return parts[2], parts[3]
        return parts[2], parts[1]
    raise ValueError(f"Cannot parse object/attribute from: {parts}")


# === Frame Section ===

def latex_frame(template: Template) -> List[str]:
    f = template.frame
    lines = [
        r"\noindent\textbf{Frame:} Our system produces $\phi=(T,O,P,V)$, where:",
        r"\begin{eqnarray*}"
    ]
    lines += latex_types(f)
    lines += latex_objects(template)
    lines += latex_concepts(f)
    lines += latex_vars(f)
    lines += [r"\end{eqnarray*}"]
    return lines

def latex_types(frame: Frame) -> List[str]:
    lines = ["T &=& \\{ " + " \\\\".join(convert_to_mathit(str(t)) for t in sorted(frame.types)) + r" \\}\\"]
    return lines

def latex_objects(template: Template) -> List[str]:
    objs = [
        f"{convert_to_mathit(o)}:\\ {convert_to_mathit(t)}"
        for o,t in template.get_objects()
        if o not in set(template.get_exogeneous_objects())
    ]
    return ["O &=& \\{ " + " \\\\".join(objs) + r" \\}\\"]

def latex_concepts(frame: Frame) -> List[str]:
    concepts = [
        f"{convert_to_mathit(c.name)}({', '.join(convert_to_mathit(t.name) for t in ts)})"
        for c,ts in get_all_concepts(frame)
    ]
    return ["P &=& \\{ " + " \\\\".join(concepts) + r" \\}\\"]

def get_all_concepts(frame: Frame) -> List[Tuple[Any, List[Any]]]:
    result = []
    for c,ts in frame.fluid_concepts:
        result.append((c, ts))
    for c,kind,ts in frame.permanent_concepts:
        if kind=='Constructed':
            result.append((c, ts))
    return sorted(result, key=lambda x: x[0].name)

def latex_vars(frame: Frame) -> List[str]:
    vars_ = [
        f"{convert_var(str(v))}: {convert_to_mathit(t.name)}"
        for v,t in frame.vars
    ]
    return ["V &=& \\{ " + " \\\\".join(vars_) + r" \\}\\"]


# === Interpretation Section ===

def latex_interpretation(template: Template, interp: Interpretation) -> List[str]:
    lines = [
        r"\noindent\textbf{Interpretation:} Our theory $\theta=(\phi,I,R,C)$ has",
        r"\begin{eqnarray*}"
    ]
    lines += latex_initial_conditions(interp)
    lines += latex_update_rules(interp)
    lines += latex_conditions(template, interp)
    lines += [r"\end{eqnarray*}"]
    return lines

def latex_initial_conditions(interp: Interpretation) -> List[str]:
    init = [convert_atom(a) for a in interp.inits + interp.permanents]
    body = r" \\ ".join(init)
    return [r"I &=& \left\{ \begin{array}{l}" + body + r"\end{array}\right\}\\"]

def latex_update_rules(interp: Interpretation) -> List[str]:
    rules = [latex_rule(r) for r in sorted(interp.rules, key=lambda r: str(r))]
    body = "".join(r + r"\\" for r in rules)
    return [r"R &=& \left\{ \begin{array}{l}" + body + r"\end{array}\right\}\\"]

def latex_rule(rule: Rule) -> str:
    connector = r"\rightarrow" if isinstance(rule, Arrow) else r"\fork"
    lhs = ' \\wedge '.join(convert_atom(b) for b in rule.body)
    rhs = convert_atom(rule.head)
    return f"{lhs} {connector} {rhs}"

def latex_conditions(template: Template, interp: Interpretation) -> List[str]:
    conds = [convert_condition(template, c) + r"\\" for c in interp.exclusions]
    # add uniqueness constraints for binary fluents
    for fname,t1,t2 in binary_fluents(template.frame):
        conds.append(
            rf"\forall X:{t1},\,\exists!Y:{t2},\,{fname}(X,Y)\\"
        )
    body = "".join(conds)
    return [r"C &=& \left\{ \begin{array}{l}" + body + r"\end{array}\right\}\\"]

def binary_fluents(frame: Frame) -> List[Tuple[str,str,str]]:
    return [
        (c.name, types[0].name, types[1].name)
        for c,types in frame.fluid_concepts if len(types)==2
    ]

def convert_condition(template: Template, s: str) -> str:
    atom = replace_atom(s)
    parts = sep2('+', atom)
    t = get_unary_predicate_type(template.frame, parts[0])
    terms = ' \\oplus '.join(f"{convert_to_mathit(x)}(X)" for x in parts)
    return rf"\forall X:{t},\,{terms}"


# === Atom/token conversion utilities ===

def replace_atom(s: str) -> str:
    """
    Strip the internal prefixes c_, p_, obj_ everywhere.
    """
    return s.replace('c_', '').replace('p_', '').replace('obj_', '')

def convert_atom(s: str) -> str:
    """
    Turn something like "in_1(x1,cell_3_1)" into \mathit{in\_1}(x1,cell\_3\_1),
    after stripping prefixes.
    """
    raw = replace_atom(s)
    m = re.match(r'(\w+)\((.*)\)', raw)
    if m:
        fun = m.group(1)
        args = [a.strip() for a in m.group(2).split(',') if a.strip()]
        args2 = ','.join(convert_term(a) for a in args)
        return f"{convert_to_mathit(fun)}({args2})"
    else:
        return convert_to_mathit(raw)

def convert_term(s: str) -> str:
    return convert_to_mathit(convert_var(s))

def convert_var(s: str) -> str:
    return s[4:].upper() if s.startswith('var_') else s

def split_alpha(s: str) -> Tuple[str,str]:
    m = re.search(r'[^A-Za-z]', s)
    return (s[:m.start()], s[m.start():]) if m else (s, '')

def convert_to_mathit(s: str) -> str:
    """
    Wrap letter-start strings in \mathit, but leave pure digits alone.
    """
    s = str(s)
    if s.isdigit() or not s:
        return s
    x, y = split_alpha(s)
    return f"\\mathit{{{x}}}{y}"

def sep2(delims: str, s: str) -> List[str]:
    clean = ''.join(' ' if c in delims else c for c in s)
    return clean.split()

def words_p(p: Any, s: str) -> List[str]:
    res, i, n = [], 0, len(s)
    while i < n:
        while i < n and p(s[i]):
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not p(s[j]):
            j += 1
        res.append(s[i:j])
        i = j
    return res

def split_one_of(delims: str, s: str) -> List[str]:
    return words_p(lambda c: c in delims, s)

def get_unary_predicate_type(frame: Frame, pred: str) -> str:
    # First check fluid_concepts
    for concept, types in frame.fluid_concepts:
        if concept.name == pred and len(types) == 1:
            return types[0].name
    # Then permanent_concepts: list of triples (concept, kind, types)
    for concept, kind, types in frame.permanent_concepts:
        if concept.name == pred and len(types) == 1:
            return types[0].name
    raise KeyError(f"Failed to find unary predicate type for {pred}")

