
import sys
import subprocess
from itertools import product, count
from dataclasses import dataclass
from typing import List, Tuple, Union, Iterator, Optional
import os
import re
import time
import logging
import argparse
import clingo
from clingo.symbol import Number, Function, Symbol
from typing import Iterable, Sequence
from Interpretation import Type as T, Object, Var, Concept, Template
from ClingoParser import ClingoOutput, ClingoResult, ClingoParser, ClingoPresenter, write_latex
import gc

from pathlib import Path


try:
    import matplotlib.pyplot as _plt
except Exception:  # pragma: no cover - optional dependency
    _plt = None

logger = logging.getLogger(__name__)

from SolveTemplates import (
    template_sokoban,
    # template_eca_small,
    template_eca,
    make_eca_template,
    template_pacman,
)
import SokobanData2D as SokobanData
import SokobanTypes
import PacmanData2D as PacmanData
import PacmanTypes


# Constants
xor_group_file_name = "temp/gen_xor_groups.lp"
flag_condor: bool = False
flag_delete_temp: bool = False
flag_output_latex: bool = False
flag_ablation_remove_cost: bool = False
flag_ablation_remove_permanents: bool = False
const_time_limit: int = 14400

show_answer_set: bool = False
show_extraction: bool = True



parser = ClingoParser(show_answer_set=show_answer_set,
                     show_extraction=show_extraction)

presenter = ClingoPresenter(show_answer_set=show_answer_set,
                            show_extraction=show_extraction,
                            flag_output_latex=flag_output_latex)

# Track timings
_step_durations: list[tuple[int, float]] = []  


# ``hint(atom) -> bool`` — whether this external is currently asserted true
_known_externals: dict[Function, bool] = {}
_guide_id = count()  # monotonically increasing suffix for #program names

BASE_PRIORITY: int = 10            # initial priority of a hint batch
PRIORITY_STEP: int = 1             # priority increment per horizon step


INTEREST: set[str] = {
    "use_rule",
    "rule_var_group",
    "rule_causes_head",
    "rule_arrow_head",
    "rule_body",
    "init",
    "gen_permanent",
    "force"
}

# ---------------------------- Sokoban-specific solving ----------------------------

def get_sokoban_data(name: str) -> Tuple[int, int, int]:
    e = next((i for i in SokobanData.sokoban_examples if i[0] == name),None)
    if not e :
        raise ValueError(f"No sokoban entry called {name}")
    return extract_sokoban_data(e[1])


def extract_sokoban_data(e: SokobanTypes.Example) -> Tuple[int, int, int]:
    s = e.initial_state  # List[str]
    max_x = len(s[0])
    max_y = len(s)
    num_blocks = sum(row.count('b') for row in s)
    return max_x, max_y, num_blocks


# ---------------------------- Pacman-specific solving -----------------------------

def get_pacman_data(name: str) -> Tuple[int, int, int, int]:
    e = next((i for i in PacmanData.pacman_examples if i[0] == name), None)
    if not e:
        raise ValueError(f"No pacman entry called {name}")
    return extract_pacman_data(e[1])


def extract_pacman_data(e: PacmanTypes.Example) -> Tuple[int, int, int, int]:
    s = e.initial_state
    max_x = len(s[0])
    max_y = len(s)
    num_pellets = sum(row.count('o') for row in s)
    num_ghosts = sum(row.count('g') for row in s)
    return max_x, max_y, num_pellets, num_ghosts


def solve_sokoban(example_name: str, incremental: bool = False) -> Optional[str]:
    """Solve a Sokoban instance and return the formatted result."""
    logger.info("Using example: %s", example_name)

    max_x, max_y, n_blocks = get_sokoban_data(example_name)
    t = template_sokoban(max_x, max_y, n_blocks)

    logger.info("max_x: %s max_y: %s n_blocks: %s", max_x, max_y, n_blocks)
    input_f = f"predict_{example_name}.lp"
    _, solutions = do_solve("data/sokoban", input_f, t, incremental=incremental)

    if not solutions:
        logger.warning("No solution found.")
        return None
    
    # print(f"Found {len(solutions)} solutions.")
    # for ans in answer:
    #     if flag_output_latex:
    #         write_latex(t, ans)
    # outputs = [presenter.present(t, ans) for ans in answer]
    result = presenter.present(t, solutions)
    logger.info("Solution found:\n%s", result)
    return result


def solve_pacman(example_name: str, incremental: bool = False) -> Optional[str]:
    """Solve a Pacman instance and return the formatted result."""

    logger.info("Using example: %s", example_name)

    max_x, max_y, num_pellets, num_ghosts = get_pacman_data(example_name)
    t = template_pacman(max_x, max_y, num_pellets, num_ghosts)

    logger.info(
        "max_x: %s max_y: %s pellets: %s ghosts: %s",
        max_x,
        max_y,
        num_pellets,
        num_ghosts,
    )
    input_f = f"predict_{example_name}.lp"
    _, solutions = do_solve("data/pacman", input_f, t, incremental=incremental)

    if not solutions:
        logger.warning("No solution found.")
        return None

    result = presenter.present(t, solutions)
    logger.info("Solution found:\n%s", result)
    return result


def solve_eca(input_name: str, incremental: bool = False) -> Optional[str]:
    """Solve an ECA instance with optional incremental solving."""

    logger.info("Using ECA input: %s", input_name)
    t = template_eca(False)
    input_f = f"{input_name}.lp" if not input_name.endswith(".lp") else input_name

    _, solutions = do_solve("data/eca", input_f, t, incremental=incremental)

    if not solutions:
        logger.warning("No solution found.")
        return None

    result = presenter.present(t, solutions)
    logger.info("Solution found:\n%s", result)
    return result



# -------------------------------------------------------------------------------
# ECA-specific iteration
# -------------------------------------------------------------------------------

# def solve_eca_iteratively(input_f):
#     """
#     Solves ECA problems iteratively using a specific set of templates.

#     Args:
#         input_f (str): The input file prefix.
#     """
#     # Haskell: solve_iteratively "data/eca" input_f (all_eca_templates input_f) False False
#     solve_iteratively("data/eca", input_f, all_eca_templates(input_f), False, False)

def all_eca_templates(input_f):
    """
    Generates all ECA templates for iterative solving.

    Args:
        input_f (str): The input file prefix. (Note: this argument is effectively
                       ignored in the Haskell `make_eca_template` call with `False`.)

    Returns:
        list: A list of (description_string, Template_object) tuples.
    """
    # Haskell: map (make_eca_template False input_f) [0..8]
    return [make_eca_template(False, input_f, i) for i in range(9)] # 0 to 8 inclusive

# # -------------------------------------------------------------------------------
# # ECA iteration using the general code for template iteration
# # -------------------------------------------------------------------------------

# def solve_eca_general(input_f):
#     """
#     Solves ECA problems using general template iteration code.

#     Args:
#         input_f (str): The input file prefix.
#     """
#     # Haskell: do solve_iteratively "data/misc" input_f (all_general_eca_templates input_f) False False
#     solve_iteratively("data/misc", input_f, all_general_eca_templates(input_f), False, False)

# def all_general_eca_templates(input_f):
#     """
#     Generates all general ECA templates by augmenting a base template.

#     Args:
#         input_f (str): The input file prefix. (Note: this argument is unused in
#                        the Haskell `all_general_eca_templates` logic.)

#     Returns:
#         list: A list of (description_string, Template_object) tuples.
#     """
#     # Haskell: f (i, t) = ("Template " ++ show i, t)
#     # Haskell: ps = parameter_lists [T "sensor"] 100
#     # Haskell: ts = map (augment_template t') ps
#     # Haskell: t' = template_eca_small
    
#     t_prime = template_eca_small
#     # Assuming Type("sensor") is defined
#     ps = parameter_lists([T("sensor")], 100)
#     ts = [augment_template(t_prime, p) for p in ps]

#     # zip [1..] ts
#     # In Python, enumerate starts from 0, so we add 1 for 1-based indexing.
#     return [(f"Template {i+1}", t) for i, t in enumerate(ts)]

# def output_general_eca_templates(input_f, n):
#     """
#     Outputs LaTeX representations of general ECA templates.

#     Args:
#         input_f (str): The input file prefix. (Unused in Haskell logic).
#         n (int): The number of templates to output.
#     """
#     # Haskell: Monad.forM_ xs f where xs = map snd $ take n (all_general_eca_templates input_f); f t = Monad.forM_ (latex_frame t) putStrLn
    
#     # xs = map snd $ take n (all_general_eca_templates input_f)
#     # In Python: get the Template objects from the generated list
#     templates_to_output = [t for _, t in all_general_eca_templates(input_f)][:n]

#     # f t = Monad.forM_ (latex_frame t) putStrLn
#     for t_obj in templates_to_output:
#         # for line in latex_frame(t_obj):
#         #     print(line)
#         continue




# ---------------------------- Iterative solving ----------------------------

# def solve_iteratively(directory: str, input_file: str, templates: List[Tuple[str, Template]], continue_flag: bool, output_intermediaries: bool) -> None:
#     solve_iteratively2(directory, input_file, templates, continue_flag, output_intermediaries, None)


# def solve_iteratively2(directory: str, input_file: str, templates: List[Tuple[str, Template]], 
#                        continue_flag: bool, output_intermediaries: bool, 
#                        best: Optional[ClingoResult]) -> None:
#     if not templates and not continue_flag:
#         logger.error("Unable to solve %s", input_file)
#         return
#     if not templates and continue_flag and best is None:
#         logger.error("Unable to solve %s", input_file)
#         return
#     if not templates and continue_flag and best is not None:
#         logger.info("Best answer:")
#         tpl = best.result_template
#         logger.info(presenter.present(tpl, ClingoOutput(answer=best.result_answer)))
#         logger.info(presenter.present(tpl, ClingoOutput(optimization=best.result_optimization)))
#         return
#     label, tpl = templates[0]
#     logger.info(label)
#     results_file, outputs = do_solve(directory, input_file, tpl)
#     if not outputs:
#         logger.info("No solution found for this configuration")
#         logger.info("")
#         solve_iteratively2(directory, input_file, templates[1:], continue_flag, output_intermediaries, best)
#         return

#     last = parser.last_outputs([outputs]) if outputs else []



#     if output_intermediaries or not continue_flag:
#         for ans in last:
#             logger.info(presenter.present(tpl, ans))
#     if not continue_flag:
#         return
#     new_best = update_best(tpl, best, last)
#     solve_iteratively2(directory, input_file, templates[1:], continue_flag, output_intermediaries, new_best)


# def update_best(template: Template, current: Optional[ClingoResult], outputs: List[ClingoOutput]) -> Optional[ClingoResult]:
#     if current is None and len(outputs) >= 2:
#         return ClingoResult(answer=outputs[0].answer, optimization=outputs[1].optimization, template=template)
#     if current is not None and len(outputs) >= 2:
#         if less_optim(outputs[1].optimization, current.result_optimization):
#             return ClingoResult(answer=outputs[0].answer, optimization=outputs[1].optimization, template=template)
#     return current


# def less_optim(x: str, y: str) -> bool:
#     xs = list(map(int, x.split()))
#     ys = list(map(int, y.split()))
#     return xs < ys


def get_num_time_steps(input_file: str) -> int:
    # use regex to extract the nuber from lines like: senses(*, 1). where 1 is the time step. find all the instances of this pattern as get the biggest value
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    
    with open(input_file, 'r') as f:
        content = f.read()
    matches = re.findall(r'senses\([^)]*\),\s*(\d+)\)\.', content)
    if not matches:
        raise ValueError(f"No time steps found in {input_file}.")
    return max(int(match) for match in matches)


def get_assumptions(model: Sequence[clingo.Symbol]) -> list:
    assumptions = []
    for atom in model:
        if atom.name in {
            "use_rule",
            "rule_var_group",
            "rule_causes_head",
            "rule_arrow_head",
            "rule_body",
        }:
            assumptions.append((Function(atom.name, atom.arguments), True))
    return assumptions

    
def model_to_string(model: Sequence[clingo.Symbol]) -> str:
    return " ".join(
        f"{atom.name}({','.join(str(arg) for arg in atom.arguments)})"
        for atom in model
    )

def make_model_cb(
    collector: list[List[Symbol]],
    step: int,

):
    """Return an *on_model* callback that stores *shown* symbols."""

    def cb(model: clingo.Model):
        logger.info("Model found with cost %s at step %s", model.cost, step)
        collector.append(model.symbols(shown=True))
        return None

    return cb

def pretty(model: Iterable[Symbol], template, parser, presenter) -> List[str]:
    """Format the model for human consumption."""

    line = " ".join(
        f"{a.name}({','.join(map(str, a.arguments))})" for a in model
    )
    return presenter.present(template, parser.parse_lines([line])[0])

def _interesting_atoms(model: Sequence[Symbol]) -> list[Symbol]:
    """Return the subset of *model* that should serve as hints / assumptions."""
    return [a for a in model if a.name in INTEREST]


def _get_rule_id(atom: Symbol) -> Symbol | None:
    """Return the rule identifier encoded in *atom* if present."""

    if atom.name in {
        "use_rule",
        "rule_var_group",
        "rule_causes_head",
        "rule_arrow_head",
        "rule_body",
    } and atom.arguments:
        return atom.arguments[0].name
    return None


def _wipe_all_hints(ctrl: clingo.Control) -> None:
    """Set every known ``hint/1`` external to *false* and remove it."""
    for ext, is_on in list(_known_externals.items()):
        if is_on:
            ctrl.release_external(ext)
            _known_externals.pop(ext, None)

def _activate_hints(ctrl: clingo.Control, atoms: Iterable[Symbol], step: int) -> None:
    """Attach *heuristic* guidance for *atoms* (declare once, enable this step)."""

    atoms = list(atoms)
    if not atoms:
        return

    priority = BASE_PRIORITY + PRIORITY_STEP * step

    new_program_lines: list[str] = []   # declarations not yet in the program
    to_enable: list[Function] = []      # externals to switch on *this* step

    for sym in atoms:
        ext = Function("hint", [sym])
        to_enable.append(ext)

        # Declare external & heuristic only once for the lifetime of *ctrl*
        if ext in _known_externals:
            continue

        new_program_lines.append(f"#external {ext}.")
        new_program_lines.append(
            f"#heuristic {sym} : {ext}. [1@{priority},true]"
        )
        _known_externals[ext] = False  # remember, but currently off

    # Ground the freshly created declarations
    if new_program_lines:
        tag = f"guide_{next(_guide_id)}"
        ctrl.add(tag, [], "\n".join(new_program_lines))
        ctrl.ground([(tag, [])])

    # Activate requested externals for this step
    for ext in to_enable:
        if not _known_externals[ext]:
            ctrl.assign_external(ext, True)
            _known_externals[ext] = True


def _make_graph(step_durations,outfile: Path) -> None:
    """Plot bar-chart of per-step solve times and a line of cumulative time."""
    if _plt is None or not _step_durations:
        return

    steps, durations = zip(*step_durations)
    cumulative = [sum(durations[:i + 1]) for i in range(len(durations))]

    _plt.figure(figsize=(max(6, len(steps) * 0.6), 4))
    _plt.bar(steps, durations, label="solve time per step (s)")
    _plt.plot(steps, cumulative, marker="o", label="cumulative runtime (s)")
    _plt.xlabel("step")
    _plt.ylabel("seconds")
    _plt.title("Solve-time profile")
    _plt.tight_layout()
    _plt.legend()
    _plt.savefig(outfile, dpi=150)
    _plt.close()

def _make_assumptions(
    ctrl: clingo.Control, atoms: Iterable[Symbol]
) -> tuple[list[tuple[int, bool]], dict[int, Symbol]]:
    """Translate *atoms* to solver literals for ``assumptions`` and return a
    mapping from solver literal to ``Symbol`` for later lookup."""

    assumptions: list[tuple[int, bool]] = []
    mapping: dict[int, Symbol] = {}
    for atom in atoms:
        sym = Function(atom.name, atom.arguments)
        try:
            lit = ctrl.symbolic_atoms[sym].literal
        except KeyError:
            # If the symbolic atom is unknown just skip it
            continue
        assumptions.append((lit, True))
        mapping[lit] = sym

    return assumptions, mapping

def _collect_asp_files(tmp_dir, name, t_desc):
    files = files_to_load(tmp_dir, name, t_desc)
    files += ["asp/judgement.lp", "asp/constraints.lp", "asp/step.lp"]
    return files

def _setup_control(files):
    ctl = clingo.Control(["--parallel-mode=4"])

    for f in files:
        ctl.load(f)
    return ctl


def _minimal_assumptions(ctl: clingo.Control, assumptions: list[Symbol], rule_groups, step, models) -> list[Symbol]:
    """
    Given an initial unsatisfiable core ``assumptions`` iterate over the rule
    groups encoded in them, dropping all literals of one group at a time and
    checking satisfiability.  When removing a group yields a model, return that
    model together with the updated assumption list instead of starting a new
    solve from scratch.
    """


    for rule_group in sorted(rule_groups, key=len):        

        test_assumptions = [assumption for assumption in assumptions if _get_rule_id(assumption[0]) not in rule_group]

        logger.info("Testing assumptions without rule_ids %s (%s remaining)", rule_group , len(test_assumptions))
        # solve under these assumptions
        ctl.solve(assumptions=test_assumptions, on_model=make_model_cb(models, step))
        ctl.cleanup()
        gc.collect()  # clear solver state


def rule_groups(atoms: Iterable[Symbol]) -> list[Symbol]:
    # filter rule groups from the results
    groups = {}
    for atom in atoms:
        if atom.name == "rule_group":
            if  groups.get(atom.arguments[0].name, None) is None:
                groups[atom.arguments[0].name] = [atom.arguments[1].name]
            else:
                groups[atom.arguments[0].name] =  groups[atom.arguments[0].name] + [atom.arguments[1].name]
    
    groups_list = [sorted([k]+ v) for k, v in groups.items()]
    unique_groups = set(tuple(g) for g in groups_list)

    return unique_groups

            

    


def do_solve(
    work_dir: str,
    input_file: str,
    template: Template,
    delete_temp: bool = False,
    max_steps: int = 14,
    incremental: bool = False
) -> Tuple[str, Optional[ClingoOutput]]:
    """
    Run a Clingo solve (static or incremental), collect models, record times per step,
    write results incrementally, and plot a timing graph.
    """
    # Prepare template and files
    tmp_dir, name, _, t_desc = do_template(False, template, work_dir, input_file)
    files = _collect_asp_files(tmp_dir, name, t_desc)
    result_path = Path("temp") / f"{tmp_dir}_{name}_results.txt"
    pretty_path = result_path.with_name(f"{result_path.stem}_ex.txt")

    # Ensure clean output
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("")
    pretty_path.write_text("")

    # Solver setup
    ctl = _setup_control(files)
    ctl.ground([("base", [])])


    # Record step times
    step_times: List[Tuple[int, float]] = []

    # Run solving
    if incremental:
        models, last_model = _run_incremental(
            ctl, template, name,
            work_dir, input_file,
            result_path, pretty_path,
            step_times
        )
    else:
        models, last_model = _run_static(
            ctl, template, name,
            max_steps,
            result_path, pretty_path,
            step_times
        )

    # Plot timing graph
    graph_path = Path("temp") / f"{tmp_dir}_{name}_timing.png"
    _make_graph(step_times, graph_path)
    logger.info("Timing graph saved to %s", graph_path)

    # Optional cleanup
    if delete_temp:
        subprocess.call(f"rm temp/{tmp_dir}_*", shell=True)

    # Parse outputs
    parsed: Optional[ClingoOutput] = None
    if last_model:
        parsed = parser.parse_lines([model_to_string(last_model)])[0]

    return str(result_path), parsed


# --- helper functions ---

def _run_incremental(
    ctl,
    template,
    work_dir: str,
    input_file: str,
    pretty_path: Path,
    step_times: List[Tuple[int, float]]
):
    models, hints = [], []
    max_ts = get_num_time_steps(f"{work_dir}/{input_file}")
    logger.info("Incremental up to %s steps", max_ts)

    for step in range(1, max_ts + 1):
        ctl.ground([("step", [Number(step)])])
        if step <= 1:
            continue

        # Time this step
        t0 = time.time()
        models.clear()

        # Hard guidance
        ctl.configuration.solve.heuristic = "None"
        assumptions, lit_map = _make_assumptions(ctl, hints) if hints else ([], {})
        logger.info("Step %02d with %d assumptions", step, len(assumptions))
        result = ctl.solve(
            assumptions=[(lit_map[a],v) for a, v in assumptions],
            on_model=make_model_cb(models, step),
        )
        
        is_sat = result.satisfiable
        del result
        gc.collect()

        if not is_sat:

            rule_groups_list = rule_groups(model) if model else []


        
            _minimal_assumptions(ctl, [(lit_map[a], v) for a, v in assumptions], rule_groups_list, step, models)
            logger.info("Remaining assumptions: %s", len(assumptions))
           
            is_sat = result.satisfiable
 
        # Fallback to soft
        if not is_sat:
            ctl.cleanup()
            models.clear()
            _activate_hints(ctl, hints, step)
            ctl.configuration.solve.heuristic = "Vsids-Domain"
            logger.info("No model found with hard guidance, using soft heuristics.")
            ctl.solve(on_model=make_model_cb(models, step))

        gc.collect()

        # Record time and result
        duration = time.time() - t0
        step_times.append((step, duration))

        if not models:
            logger.info("No model at step %s, stopping.", step)
            break

        model = models[-1]
        hints = _interesting_atoms(model)
        if step % 5 == 0:
            _wipe_all_hints(ctl)
        ctl.cleanup()
    



        with pretty_path.open("a") as file:
            file.write(f"--- Model {step:02d} ---\n")
            file.writelines(pretty(model, template, parser, presenter))
            file.write("\n\n")

        logger.info("Step %02d: %.2fs, written to results.", step, duration)

    return models, (models[-1] if models else None)


def _run_static(
    ctl,
    template,
    name: str,
    max_steps: int,
    result_path: Path,
    pretty_path: Path,
    step_times: List[Tuple[int, float]]
):
    models = []
    t0 = time.time()
    for step in range(1, max_steps + 1):
        ctl.ground([("step", [Number(step)])])
    ctl.solve(on_model=make_model_cb(models, step, template, parser, presenter, name))

    duration = time.time() - t0
    step_times.append((max_steps, duration))

    if not models:
        logger.info("No model found.")
        return [], None

    model = models[-1]

    with pretty_path.open("a") as file:
        file.write(pretty(model, template, parser, presenter))

    logger.info("Static solve: %.2fs for %s steps.", duration, max_steps)
    return models, model


def do_template(add_const: bool, t: Template, dir: str, input_f: str) -> Tuple[str, str, bool, Template]:
    """Generate auxiliary files for *t* and return metadata."""

    d = dir[len("data/") :]
    input_name = input_f[:-len(".lp")]
    name = f"{d}_{input_name}"
    os.makedirs("temp", exist_ok=True)
    t.gen_inits(name)
    t.gen_subs(name)
    t.gen_var_atoms(name)
    t.gen_interpretation(name)
    return d, input_name, add_const, t

def files_to_load(dir: str, input_f: str,t:Template) -> List[str]:
    task_file = f"data/{dir}/{input_f}.lp"
    d = "temp/"
    init_f = f"{d}{dir}_{input_f}_init.lp"
    subs_f = f"{d}{dir}_{input_f}_subs.lp"
    rules_f = f"{d}{dir}_{input_f}_var_atoms.lp"
    interp_f = f"{d}{dir}_{input_f}_interpretation.lp"
    auxs = [f"asp/{x}" for x in t.frame.aux_files]
    return [task_file, init_f, subs_f, rules_f, interp_f] + auxs

def gen_bash(dir: str, input_f: str, add_const: bool, t: Template) -> Tuple[str, str]:
    name = f"{dir}_{input_f}"
    task_file = f"data/{dir}/{input_f}.lp"
    d = "temp/"
    fpath = f"{d}{name}_script.sh"
    with open(fpath, "w") as fh:
        fh.write(f'echo "Processing {task_file}."\n\n')
    init_f = f"{d}{name}_init.lp"
    subs_f = f"{d}{name}_subs.lp"
    rules_f = f"{d}{name}_var_atoms.lp"
    interp_f = f"{d}{name}_interpretation.lp"
    auxs = [f"asp/{x}" for x in t.frame.aux_files]
    aux_s = " ".join(auxs)
    results_f = f"{d}{name}_results.txt"
    handle = f" > {results_f}"
    args = f" --stats --verbose=2 --warn=no-atom-undefined --time-limit={const_time_limit} "
    args_prime = args + f"-c k_xor_group=$1 {xor_group_file_name} " if add_const else args
    clingo = "/vol/lab/clingo5/clingo " if flag_condor else "clingo "
    costs = "" if flag_ablation_remove_cost else " asp/costs.lp "
    s = (
        clingo + args_prime + task_file + " " + init_f + " " + subs_f + " " +
        rules_f + " " + interp_f + " " + aux_s +
        " asp/judgement.lp asp/constraints.lp" + costs + handle + "\n\n"
    )
    with open(fpath, "a") as fh:
        fh.write(s)
    logger.info("Generated %s", fpath)
    os.chmod(fpath, 0o777)
    return fpath, results_f


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the command line interface."""

    parser = argparse.ArgumentParser(description="Solve problems using Clingo")
    sub = parser.add_subparsers(dest="command", required=True)

    sok_cmd = sub.add_parser("sokoban", help="Solve a Sokoban instance")
    sok_cmd.add_argument("example")
    sok_cmd.add_argument("--incremental", action="store_true")

    pac_cmd = sub.add_parser("pacman", help="Solve a Pacman instance")
    pac_cmd.add_argument("example")
    pac_cmd.add_argument("--incremental", action="store_true")

    eca_cmd = sub.add_parser("eca", help="Solve an ECA instance")
    eca_cmd.add_argument("input")
    eca_cmd.add_argument("--incremental", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    if args.command == "sokoban":
        solve_sokoban(args.example, args.incremental)
    elif args.command == "pacman":
        solve_pacman(args.example, args.incremental)
    elif args.command == "eca":
        solve_eca(args.input, args.incremental)

if __name__ == "__main__":
    main()
