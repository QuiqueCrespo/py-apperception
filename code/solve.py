import argparse
import gc
import logging
import os
import re
import time
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import clingo
from clingo.symbol import Function, Number, Symbol

# Local application imports
from ClingoParser import ClingoOutput, ClingoParser, ClingoPresenter
from Interpretation import Concept, Object, Template, Type as T, Var
from SolveTemplates import (
    make_eca_template,
    template_eca,
    template_pacman,
    template_sokoban,
)
import PacmanData2D as PacmanData
import PacmanTypes
import SokobanData2D as SokobanData
import SokobanTypes
import matplotlib.pyplot as plt


# --- Configuration and Constants ---

logger = logging.getLogger(__name__)

# File Paths
XOR_GROUP_FILE_NAME = "temp/gen_xor_groups.lp"

# Flags
FLAG_CONDOR: bool = False
FLAG_DELETE_TEMP: bool = False
FLAG_OUTPUT_LATEX: bool = False
FLAG_ABLATION_REMOVE_COST: bool = False
FLAG_ABLATION_REMOVE_PERMANENTS: bool = False

# Solving Parameters
DEFAULT_TIME_LIMIT: int = 14400  # seconds
BASE_PRIORITY: int = 10  # initial priority of a hint batch
PRIORITY_STEP: int = 1  # priority increment per horizon step
MAX_INCREMENTAL_STEPS: int = 14

# Display Options
SHOW_ANSWER_SET: bool = False
SHOW_EXTRACTION: bool = True

# Clingo Atom Interests
INTERESTING_ATOMS: set[str] = {
    "use_rule",
    "rule_var_group",
    "rule_causes_head",
    "rule_arrow_head",
    "rule_body",
    "init",
    "gen_permanent",
    "force",
}

# --- Global State (Carefully managed) ---

# `hint(atom) -> bool` — whether this external is currently asserted true
_known_externals: dict[Function, bool] = {}
_guide_id_counter = count()  # monotonically increasing suffix for #program names
_step_durations: list[tuple[int, float]] = []  # Track timings


# --- Clingo Presenter and Parser Initialization ---

parser = ClingoParser(show_answer_set=SHOW_ANSWER_SET, show_extraction=SHOW_EXTRACTION)
presenter = ClingoPresenter(
    show_answer_set=SHOW_ANSWER_SET,
    show_extraction=SHOW_EXTRACTION,
    flag_output_latex=FLAG_OUTPUT_LATEX,
)


# --- Data Extraction Functions ---

def get_sokoban_data(name: str) -> Tuple[int, int, int]:
    """
    Retrieves and extracts Sokoban specific data (max_x, max_y, num_blocks)
    for a given example name.

    Args:
        name: The name of the Sokoban example.

    Returns:
        A tuple containing (max_x, max_y, num_blocks).

    Raises:
        ValueError: If no Sokoban entry with the given name is found.
    """
    example_entry = next(
        (i for i in SokobanData.sokoban_examples if i[0] == name), None
    )
    if not example_entry:
        raise ValueError(f"No Sokoban entry called {name} found.")
    return _extract_sokoban_properties(example_entry[1])


def _extract_sokoban_properties(example: SokobanTypes.Example) -> Tuple[int, int, int]:
    """
    Extracts maximum x, maximum y, and number of blocks from a Sokoban example.

    Args:
        example: The SokobanTypes.Example object.

    Returns:
        A tuple containing (max_x, max_y, num_blocks).
    """
    initial_state = example.initial_state  # List[str]
    max_x = len(initial_state[0])
    max_y = len(initial_state)
    num_blocks = sum(row.count("b") for row in initial_state)
    return max_x, max_y, num_blocks


def get_pacman_data(name: str) -> Tuple[int, int, int, int]:
    """
    Retrieves and extracts Pacman specific data (max_x, max_y, num_pellets, num_ghosts)
    for a given example name.

    Args:
        name: The name of the Pacman example.

    Returns:
        A tuple containing (max_x, max_y, num_pellets, num_ghosts).

    Raises:
        ValueError: If no Pacman entry with the given name is found.
    """
    example_entry = next((i for i in PacmanData.pacman_examples if i[0] == name), None)
    if not example_entry:
        raise ValueError(f"No Pacman entry called {name} found.")
    return _extract_pacman_properties(example_entry[1])


def _extract_pacman_properties(
    example: PacmanTypes.Example,
) -> Tuple[int, int, int, int]:
    """
    Extracts maximum x, maximum y, number of pellets, and number of ghosts
    from a Pacman example.

    Args:
        example: The PacmanTypes.Example object.

    Returns:
        A tuple containing (max_x, max_y, num_pellets, num_ghosts).
    """
    initial_state = example.initial_state
    max_x = len(initial_state[0])
    max_y = len(initial_state)
    num_pellets = sum(row.count("o") for row in initial_state)
    num_ghosts = sum(row.count("g") for row in initial_state)
    return max_x, max_y, num_pellets, num_ghosts


def get_num_time_steps(input_file_path: str) -> int:
    """
    Extracts the maximum time step from an input ASP file by finding
    the largest number in `senses(*, T)` patterns.

    Args:
        input_file_path: The path to the input ASP file.

    Returns:
        The maximum time step found in the file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If no time steps are found in the file.
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file {input_file_path} does not exist.")

    with open(input_file_path, "r") as f:
        content = f.read()

    # Regex to find numbers in senses(*, 1). where 1 is the time step
    matches = re.findall(r"senses\([^)]*\),\s*(\d+)\)\.", content)
    if not matches:
        raise ValueError(f"No time steps found in {input_file_path}.")

    return max(int(match) for match in matches)


# --- Clingo Interaction Helper Functions ---


def _interesting_atoms(model: Sequence[Symbol]) -> List[Symbol]:
    """Return the subset of *model* that should serve as hints / assumptions."""
    return [a for a in model if a.name in INTERESTING_ATOMS]


def _get_rule_id(atom: Symbol) -> Optional[str]:
    """Return the rule identifier encoded in *atom* if present."""
    if atom.name in INTERESTING_ATOMS and atom.arguments:
        # Assuming the rule ID is the first argument and is a name (string)
        return str(atom.arguments[0])
    return None


def _wipe_all_hints(ctrl: clingo.Control) -> None:
    """Set every known `hint/1` external to `false` and remove it."""
    for ext, is_on in list(_known_externals.items()):
        if is_on:
            ctrl.release_external(ext)
            _known_externals.pop(ext, None)


def _activate_hints(ctrl: clingo.Control, atoms: Iterable[Symbol], step: int) -> None:
    """
    Attach heuristic guidance for atoms (declare once, enable this step).
    New externals are declared and grounded; existing ones are assigned true.
    """
    atoms = list(atoms)
    if not atoms:
        return

    priority = BASE_PRIORITY + PRIORITY_STEP * step

    new_program_lines: list[str] = []  # declarations not yet in the program
    to_enable: list[Function] = []  # externals to switch on *this* step

    for sym in atoms:
        ext = Function("hint", [sym])
        to_enable.append(ext)

        # Declare external & heuristic only once for the lifetime of `ctrl`
        if ext in _known_externals:
            continue

        new_program_lines.append(f"#external {ext}.")
        new_program_lines.append(f"#heuristic {sym} : {ext}. [1@{priority},true]")
        _known_externals[ext] = False  # remember, but currently off

    # Ground the freshly created declarations
    if new_program_lines:
        tag = f"guide_{next(_guide_id_counter)}"
        ctrl.add(tag, [], "\n".join(new_program_lines))
        ctrl.ground([(tag, [])])

    # Activate requested externals for this step
    for ext in to_enable:
        if not _known_externals[ext]:
            ctrl.assign_external(ext, True)
            _known_externals[ext] = True


def _make_assumptions(
    ctl: clingo.Control, atoms: Iterable[Symbol]
) -> Tuple[List[Tuple[int, bool]], dict[int, Symbol]]:
    """
    Translates Clingo `Symbol` objects into solver literals for `assumptions`
    and returns a mapping from solver literal to `Symbol` for later lookup.

    Args:
        ctl: The Clingo Control object.
        atoms: An iterable of Clingo Symbol objects to be used as assumptions.

    Returns:
        A tuple containing:
            - A list of (literal, boolean_value) tuples for Clingo's `solve` method.
            - A dictionary mapping solver literals (integers) to their
              corresponding Clingo Symbol objects.
    """
    assumptions: List[Tuple[int, bool]] = []
    literal_to_symbol_map: dict[int, Symbol] = {}

    for atom in atoms:
        sym = Function(atom.name, atom.arguments)
        try:
            literal = ctl.symbolic_atoms[sym].literal
        except KeyError:
            # If the symbolic atom is unknown, skip it
            logger.debug("Skipping unknown symbolic atom: %s", sym)
            continue
        assumptions.append((literal, True))
        literal_to_symbol_map[literal] = sym

    return assumptions, literal_to_symbol_map


def _collect_asp_files(temp_dir: str, name: str, template: Template) -> List[str]:
    """
    Collects all necessary ASP files for loading into Clingo.

    Args:
        temp_dir: The temporary directory prefix for generated files.
        name: The base name for generated files.
        template: The Template object containing auxiliary files.

    Returns:
        A list of file paths to be loaded by Clingo.
    """
    task_file = f"data/{temp_dir}/{name}.lp"
    temp_prefix = f"temp/{temp_dir}_{name}"
    init_file = f"{temp_prefix}_init.lp"
    subs_file = f"{temp_prefix}_subs.lp"
    rules_file = f"{temp_prefix}_var_atoms.lp"
    interp_file = f"{temp_prefix}_interpretation.lp"
    aux_files = [f"asp/{x}" for x in template.frame.aux_files]

    return [task_file, init_file, subs_file, rules_file, interp_file] + aux_files


def _setup_control(files: List[str]) -> clingo.Control:
    """
    Initializes a Clingo Control object and loads specified ASP files.

    Args:
        files: A list of file paths to load.

    Returns:
        A configured Clingo Control object.
    """
    ctl = clingo.Control([ "--parallel-mode=2","--keep-facts", "--opt-mode=optN"])
    for f in files:
        ctl.load(f)
    return ctl


def model_to_string(model: Sequence[clingo.Symbol]) -> str:
    """
    Converts a sequence of Clingo Symbols (a model) into a space-separated string
    representation.
    """
    return " ".join(
        f"{atom.name}({','.join(str(arg) for arg in atom.arguments)})"
        for atom in model
    )


@dataclass
class ModelCost:
    cost: Sequence[int]
    symbols: List[Symbol]


def make_model_callback(
    collector: List[List[Symbol]],
    step: int,
    cost_collector: Optional[List[ModelCost]] = None,
) -> callable:
    """Return an ``on_model`` callback that records shown symbols and cost."""

    def callback(model: clingo.Model):
        logger.info("Model found with cost %s at step %s", model.cost, step)
        syms = model.symbols(shown=True)
        collector.append(syms)
        if cost_collector is not None:
            cost_collector.append(ModelCost(tuple(model.cost), list(syms)))

    return callback


def pretty_print_model(
    model: Iterable[Symbol], template: Template
) -> List[str]:
    """
    Formats the model for human consumption using the configured presenter.

    Args:
        model: An iterable of Clingo Symbols representing the model.
        template: The Template object associated with the problem.

    Returns:
        A list of formatted strings representing the model.
    """
    model_str = model_to_string(list(model))
    # ClingoParser expects a list of lines, even if it's just one model string
    return presenter.present(template, parser.parse_lines([model_str])[0])


def _make_timing_graph(step_durations: List[Tuple[int, float]], output_path: Path) -> None:
    """
    Plots a bar-chart of per-step solve times and a line of cumulative time.

    Args:
        step_durations: A list of (step, duration) tuples.
        output_path: The path to save the generated plot.
    """
    if plt is None or not step_durations:
        logger.warning("matplotlib not available or no step durations to plot.")
        return

    steps, durations = zip(*step_durations)
    cumulative = [sum(durations[:i + 1]) for i in range(len(durations))]

    plt.figure(figsize=(max(6, len(steps) * 0.6), 4))
    plt.bar(steps, durations, label="solve time per step (s)")
    plt.plot(steps, cumulative, marker="o", label="cumulative runtime (s)")
    plt.xlabel("Step")
    plt.ylabel("Seconds")
    plt.title("Solve-time Profile")
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _get_rule_groups(atoms: Iterable[Symbol]) -> List[Tuple[str, ...]]:
    """
    Extracts unique sorted rule groups from a collection of Clingo Symbols.

    Args:
        atoms: An iterable of Clingo Symbols.

    Returns:
        A list of tuples, where each tuple represents a sorted unique rule group.
    """
    groups_dict: dict[str, List[str]] = {}
    for atom in atoms:
        if atom.name == "rule_group" and len(atom.arguments) >= 2:
            group_id = str(atom.arguments[0])
            rule_name = str(atom.arguments[1])
            if group_id not in groups_dict:
                groups_dict[group_id] = []
            groups_dict[group_id].append(rule_name)

    # Sort rules within each group and then create unique sorted tuples
    groups_list = [tuple(sorted([k] + v)) for k, v in groups_dict.items()]
    # Convert to set and back to list to ensure uniqueness and then sort for consistent order
    return sorted(list(set(groups_list)))


def _minimal_assumptions_refinement(
    ctl: clingo.Control,
    initial_assumptions: List[Tuple[int, bool]],
    rule_groups_list: List[Tuple[str, ...]],
    step: int,
    models_collector: List[List[Symbol]],
    literal_to_symbol_map: dict[int, Symbol],
) -> bool:
    """
    Attempts to find a minimal satisfiable subset of assumptions by iteratively
    removing rule groups from an unsatisfiable core.

    Args:
        ctl: The Clingo Control object.
        initial_assumptions: The list of assumptions (literal, boolean) that led to unsatisfiability.
        rule_groups_list: A list of unique rule groups identified from the model.
        step: The current solving step.
        models_collector: The list to collect found models.
        literal_to_symbol_map: A mapping from literals to Clingo Symbols.

    Returns:
        True if a satisfiable model is found after refinement, False otherwise.
    """
    logger.info("Attempting assumption refinement...")
    for rule_group_tuple in sorted(rule_groups_list, key=len):
        # Convert rule group tuple back to string for comparison with rule IDs
        rule_group_set = set(rule_group_tuple)

        # Filter out assumptions belonging to the current rule_group_set
        test_assumptions = []
        for lit, val in initial_assumptions:
            symbol = literal_to_symbol_map.get(lit)
            if symbol and _get_rule_id(symbol) not in rule_group_set:
                test_assumptions.append((symbol, val))

        logger.info(
            "Testing assumptions without rule_ids from group %s (%s remaining)",
            rule_group_set,
            len(test_assumptions),
        )

        models_collector.clear() # Clear models for this refinement attempt
        result = ctl.solve(
            assumptions=test_assumptions, on_model=make_model_callback(models_collector, step)
        )
        is_sat = result.satisfiable
        del result
        gc.collect()

        if is_sat:
            logger.info("Satisfiable model found after removing rule group: %s", rule_group_set)
            return True
    logger.info("No satisfiable model found after trying all rule group removals.")
    return False


def _run_incremental_solve(
    ctl: clingo.Control,
    template: Template,
    work_dir: str,
    input_file: str,
    pretty_path: Path,
    step_times: List[Tuple[int, float]],
    opt_models_path: Optional[Path] = None,
) -> Tuple[List[List[Symbol]], Optional[List[Symbol]]]:
    """
    Executes an incremental Clingo solve process.

    Args:
        ctl: The Clingo Control object.
        template: The Template object for the problem.
        work_dir: The working directory for input files.
        input_file: The name of the input file.
        pretty_path: Path to write pretty-formatted models.
        step_times: List to record (step, duration) for each solve step.
        opt_models_path: Optional path to record all optimal-cost models per step.

    Returns:
        A tuple containing:
            - A list of all models found (each model being a list of symbols).
            - The last model found, or None if no solution is found.
    """
    all_models: List[List[Symbol]] = []
    current_hints: List[Symbol] = []
    max_time_steps = get_num_time_steps(f"{work_dir}/{input_file}")
    logger.info("Incremental solving up to %s steps", max_time_steps)

    if opt_models_path is not None:
        opt_models_path.write_text("")

    for step in range(1, max_time_steps + 1):
        ctl.ground([("step", [Number(step)])])

        # If it's the first step (step=0 or 1), we don't apply hints from a previous model
        if step > 1 and current_hints:
            _activate_hints(ctl, current_hints, step)
        else:
            # For the first step or if no hints, ensure heuristics are default
            ctl.configuration.solve.heuristic = "Vsids-Domain"

        t0 = time.time()
        models_current_step: List[List[Symbol]] = []
        cost_models_current_step: List[ModelCost] = []

        # Attempt to solve with hard guidance (assumptions) if hints are available
        assumptions, literal_to_symbol_map = _make_assumptions(ctl, current_hints)
        result = ctl.solve(
            assumptions=[(literal_to_symbol_map[lit], val) for lit, val in assumptions],
            on_model=make_model_callback(models_current_step, step, cost_models_current_step),
        )
        is_satisfiable = result.satisfiable
        del result  # Release solver resources early
        gc.collect()

        if not is_satisfiable and current_hints:
            logger.info("Unsatisfiable with hard guidance at step %s. Attempting refinement.", step)
            # Try to find a minimal set of assumptions by removing rule groups
            # We need the model from the previous step to identify rule groups
            # If `models_current_step` is empty here, it means no model was found to generate hints from.
            # We can use the last model found in `all_models` if it exists.
            last_successful_model = all_models[-1] if all_models else []
            rule_groups_from_last_model = _get_rule_groups(last_successful_model)

            refinement_successful = _minimal_assumptions_refinement(
                ctl,
                assumptions,
                rule_groups_from_last_model,
                step,
                models_current_step,
                literal_to_symbol_map,
            )
            is_satisfiable = refinement_successful # Update satisfiability based on refinement attempt

        if not is_satisfiable:
            logger.warning("No solution found at step %s after all attempts. Stopping incremental solve.", step)
            break

        # Record time and process result
        duration = time.time() - t0
        step_times.append((step, duration))

        if not models_current_step:
            logger.info("No model found at step %s, stopping.", step)
            break

        current_model = models_current_step[-1]
        all_models.extend(models_current_step)
        current_hints = _interesting_atoms(current_model)

        # Periodically wipe hints to prevent accumulation and potential performance issues
        if step % 5 == 0:
            _wipe_all_hints(ctl)
        ctl.cleanup()  # Clean up internal solver state

        with pretty_path.open("a") as file:
            file.write(f"--- Model {step:02d} ---\n")
            file.writelines(pretty_print_model(current_model, template))
            file.write("\n\n")

        if opt_models_path is not None and cost_models_current_step:
            best_cost = min(mc.cost for mc in cost_models_current_step)
            with opt_models_path.open("a") as opt_file:
                opt_file.write(f"--- Step {step:02d} (cost {best_cost}) ---\n")
                for mc in cost_models_current_step:
                    if mc.cost == best_cost:
                        opt_file.writelines(pretty_print_model(mc.symbols, template))
                        opt_file.write("\n\n")

        logger.info("Step %02d: %.2fs. Results written to %s", step, duration, pretty_path)

    return all_models, (all_models[-1] if all_models else None)


def _run_static_solve(
    ctl: clingo.Control,
    template: Template,
    max_steps: int,
    pretty_path: Path,
    step_times: List[Tuple[int, float]],
) -> Tuple[List[List[Symbol]], Optional[List[Symbol]]]:
    """
    Executes a static Clingo solve process.

    Args:
        ctl: The Clingo Control object.
        template: The Template object for the problem.
        max_steps: The maximum number of steps for grounding.
        pretty_path: Path to write pretty-formatted models.
        step_times: List to record (max_steps, duration) for the solve.

    Returns:
        A tuple containing:
            - A list of all models found (each model being a list of symbols).
            - The last model found, or None if no solution is found.
    """
    all_models: List[List[Symbol]] = []
    t0 = time.time()

    # Ground all steps upfront for static solving
    for step_num in range(1, max_steps + 1):
        ctl.ground([("step", [Number(step_num)])])

    ctl.solve(on_model=make_model_callback(all_models, max_steps))

    duration = time.time() - t0
    step_times.append((max_steps, duration))

    if not all_models:
        logger.info("No model found for static solve.")
        return [], None

    last_model = all_models[-1]
    with pretty_path.open("a") as file:
        file.writelines(pretty_print_model(last_model, template))

    logger.info("Static solve: %.2fs for %s steps. Results written to %s", duration, max_steps, pretty_path)
    return all_models, last_model


# --- Solving Entry Points ---


def solve(
    problem_type: str,
    example_name: str,
    incremental: bool = False,
    save_opt_models: bool = False,
) -> Optional[str]:
    """
    Solves a problem (Sokoban, Pacman, or ECA) using Clingo.

    Args:
        problem_type: "sokoban", "pacman", or "eca".
        example_name: The name of the example or input file.
        incremental: Whether to use incremental solving.
        save_opt_models: If True, write optimal-cost models per step to a file.

    Returns:
        The formatted solution as a string, or None if no solution is found.
    """
    logger.info("Solving %s problem for example: %s (incremental: %s)", problem_type, example_name, incremental)

    template: Template
    work_dir: str
    input_file: str

    if problem_type == "sokoban":
        max_x, max_y, n_blocks = get_sokoban_data(example_name)
        template = template_sokoban(max_x, max_y, n_blocks)
        work_dir = "data/sokoban"
        input_file = f"predict_{example_name}.lp"
        logger.info("Sokoban parameters: max_x=%s, max_y=%s, n_blocks=%s", max_x, max_y, n_blocks)
    elif problem_type == "pacman":
        max_x, max_y, num_pellets, num_ghosts = get_pacman_data(example_name)
        template = template_pacman(max_x, max_y, num_pellets, num_ghosts)
        work_dir = "data/pacman"
        input_file = f"predict_{example_name}.lp"
        logger.info(
            "Pacman parameters: max_x=%s, max_y=%s, pellets=%s, ghosts=%s",
            max_x, max_y, num_pellets, num_ghosts,
        )
    elif problem_type == "eca":
        template = template_eca(False) # `False` likely indicates a specific variant or setting for ECA
        work_dir = "data/eca"
        input_file = f"{example_name}.lp" if not example_name.endswith(".lp") else example_name
        logger.info("ECA input file: %s", input_file)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    # Prepare template and generate auxiliary files
    temp_dir_prefix = work_dir.split("/")[-1] # e.g., "sokoban", "pacman", "eca"
    base_name = input_file.replace(".lp", "")
    _generate_template_files(template, temp_dir_prefix, base_name)

    files_to_load = _collect_asp_files(temp_dir_prefix, base_name, template)
    # Add common ASP files
    files_to_load.extend(["asp/judgement.lp", "asp/constraints.lp", "asp/step.lp"])

    result_path = Path("temp") / f"{temp_dir_prefix}_{base_name}_results.txt"
    pretty_path = result_path.with_name(f"{result_path.stem}_ex.txt")

    # Ensure output directories exist and files are clean
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("")
    pretty_path.write_text("")

    opt_models_path = (
        result_path.with_name(f"{result_path.stem}_opt.txt")
        if incremental and save_opt_models
        else None
    )
    if opt_models_path is not None:
        opt_models_path.write_text("")

    ctl = _setup_control(files_to_load)
    ctl.ground([("base", [])])

    _step_durations.clear() # Clear global durations list for each new solve call

    models: List[List[Symbol]]
    last_model: Optional[List[Symbol]]

    if incremental:
        models, last_model = _run_incremental_solve(
            ctl,
            template,
            work_dir,
            input_file,
            pretty_path,
            _step_durations,
            opt_models_path,
        )
    else:
        models, last_model = _run_static_solve(
            ctl, template, MAX_INCREMENTAL_STEPS, pretty_path, _step_durations
        )

    # Plot timing graph
    graph_path = Path("temp") / f"{temp_dir_prefix}_{base_name}_timing.png"
    _make_timing_graph(_step_durations, graph_path)
    logger.info("Timing graph saved to %s", graph_path)

    # Optional cleanup of temporary files
    if FLAG_DELETE_TEMP:
        for f in Path("temp").glob(f"{temp_dir_prefix}_{base_name}_*"):
            f.unlink(missing_ok=True)
        logger.info("Temporary files deleted for %s/%s", temp_dir_prefix, base_name)

    if not last_model:
        logger.warning("No solution found for %s problem: %s", problem_type, example_name)
        return None

    result_str = presenter.present(template, parser.parse_lines([model_to_string(last_model)])[0])
    logger.info("Solution found:\n%s", result_str)
    return result_str


def _generate_template_files(template: Template, temp_dir: str, input_name: str) -> None:
    """
    Generates auxiliary files needed for the Clingo solver from the template.

    Args:
        template: The Template object.
        temp_dir: The directory prefix for generated files (e.g., "sokoban").
        input_name: The base name of the input file (without .lp extension).
    """
    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    full_name = f"{temp_dir}_{input_name}"
    template.gen_inits(full_name)
    template.gen_subs(full_name)
    template.gen_var_atoms(full_name)
    template.gen_interpretation(full_name)
    logger.debug("Generated template files for %s", full_name)


def all_eca_templates(input_f: str):
    """
    Generates all ECA templates for iterative solving.

    Args:
        input_f (str): The input file prefix. (Note: this argument is effectively
                       ignored in the Haskell `make_eca_template` call with `False`.)

    Returns:
        list: A list of (description_string, Template_object) tuples.
    """
    # Haskell: map (make_eca_template False input_f) [0..8]
    # The `input_f` parameter is ignored in the original `make_eca_template` call.
    return [make_eca_template(False, input_f, i) for i in range(9)]  # 0 to 8 inclusive


# --- Main Entry Point ---

def main(argv: Optional[List[str]] = None) -> None:
    """
    Entry point for the command line interface.
    """
    parser_cli = argparse.ArgumentParser(description="Solve problems using Clingo")
    subparsers = parser_cli.add_subparsers(dest="command", required=True)

    sok_cmd = subparsers.add_parser("sokoban", help="Solve a Sokoban instance")
    sok_cmd.add_argument("example", help="Name of the Sokoban example")
    sok_cmd.add_argument(
        "--incremental", action="store_true", help="Use incremental solving"
    )
    sok_cmd.add_argument(
        "--save-opt-models",
        action="store_true",
        help="Save optimal-cost models for each step",
    )

    pac_cmd = subparsers.add_parser("pacman", help="Solve a Pacman instance")
    pac_cmd.add_argument("example", help="Name of the Pacman example")
    pac_cmd.add_argument(
        "--incremental", action="store_true", help="Use incremental solving"
    )
    pac_cmd.add_argument(
        "--save-opt-models",
        action="store_true",
        help="Save optimal-cost models for each step",
    )

    eca_cmd = subparsers.add_parser("eca", help="Solve an ECA instance")
    eca_cmd.add_argument("input", help="Name of the ECA input file (e.g., 'rules')")
    eca_cmd.add_argument(
        "--incremental", action="store_true", help="Use incremental solving"
    )
    eca_cmd.add_argument(
        "--save-opt-models",
        action="store_true",
        help="Save optimal-cost models for each step",
    )

    args = parser_cli.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "sokoban":
        solve(
            problem_type="sokoban",
            example_name=args.example,
            incremental=args.incremental,
            save_opt_models=args.save_opt_models,
        )
    elif args.command == "pacman":
        solve(
            problem_type="pacman",
            example_name=args.example,
            incremental=args.incremental,
            save_opt_models=args.save_opt_models,
        )
    elif args.command == "eca":
        solve(
            problem_type="eca",
            example_name=args.input,
            incremental=args.incremental,
            save_opt_models=args.save_opt_models,
        )


if __name__ == "__main__":
    main()


# def gen_bash(dir: str, input_f: str, add_const: bool, t: Template) -> Tuple[str, str]:
#     name = f"{dir}_{input_f}"
#     task_file = f"data/{dir}/{input_f}.lp"
#     d = "temp/"
#     fpath = f"{d}{name}_script.sh"
#     with open(fpath, "w") as fh:
#         fh.write(f'echo "Processing {task_file}."\n\n')
#     init_f = f"{d}{name}_init.lp"
#     subs_f = f"{d}{name}_subs.lp"
#     rules_f = f"{d}{name}_var_atoms.lp"
#     interp_f = f"{d}{name}_interpretation.lp"
#     auxs = [f"asp/{x}" for x in t.frame.aux_files]
#     aux_s = " ".join(auxs)
#     results_f = f"{d}{name}_results.txt"
#     handle = f" > {results_f}"
#     args = f" --stats --verbose=2 --warn=no-atom-undefined --time-limit={const_time_limit} "
#     args_prime = args + f"-c k_xor_group=$1 {xor_group_file_name} " if add_const else args
#     clingo = "/vol/lab/clingo5/clingo " if flag_condor else "clingo "
#     costs = "" if flag_ablation_remove_cost else " asp/costs.lp "
#     s = (
#         clingo + args_prime + task_file + " " + init_f + " " + subs_f + " " +
#         rules_f + " " + interp_f + " " + aux_s +
#         " asp/judgement.lp asp/constraints.lp" + costs + handle + "\n\n"
#     )
#     with open(fpath, "a") as fh:
#         fh.write(s)
#     logger.info("Generated %s", fpath)
#     os.chmod(fpath, 0o777)
#     return fpath, results_f
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