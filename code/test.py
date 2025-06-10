#!/usr/bin/env python3
"""
Incremental Sokoban predictor with assumption-based warm starts.

• Re-grounds one `step/1` block per loop iteration.
• Carries over the “interesting” atoms from the previous model as assumptions.
• Logs concise statistics for every solve call and at the end.

Tested with clingo 5.6 (release) and 5.7 (clingox); the statistics callback
adapts automatically to either solve-API signature.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import clingo
from clingo.symbol import Function, Number

from ClingoParser import ClingoParser, ClingoPresenter
from SolveTemplates import template_sokoban

# ------------------------------------------------------------------------------
# configuration
# ------------------------------------------------------------------------------

FILES: tuple[str, ...] = (
    "data/sokoban/predict_e1d_0_0.lp",
    "temp/sokoban_predict_e1d_0_0_init.lp",
    "temp/sokoban_predict_e1d_0_0_subs.lp",
    "temp/sokoban_predict_e1d_0_0_var_atoms.lp",
    "temp/sokoban_predict_e1d_0_0_interpretation.lp",
    "asp/judgement.lp",
    "asp/constraints.lp",
    "asp/step.lp",
)

INTEREST_PREDICATES: set[str] = {
    "use_rule",
    "rule_var_group",
    "rule_causes_head",
    "rule_arrow_head",
    "rule_body",
}

MAX_STEPS = 14
INITIAL_GRID = (5, 1, 0)  #  (width, height, #boxes)

LOG_FMT = "[%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------


def extract_assumptions(model: Sequence[clingo.Symbol]) -> List[Tuple[Function, bool]]:
    """Return the (literal, truth) pairs to be used as assumptions."""
    return [
        (Function(atom.name, atom.arguments), True)
        for atom in model
        if atom.name in INTEREST_PREDICATES
    ]


def make_on_model(storage: list[list[clingo.Symbol]]) -> Callable[[clingo.Model], None]:
    """Factory: capture models in *storage* (last one at index -1)."""

    def _cb(model: clingo.Model) -> None:  # noqa: D401
        storage.append(model.symbols(shown=True))

    return _cb


def make_on_stats(step_no: int) -> Callable[..., None]:
    """
    Statistics callback that works for both clingo ≤ 5.6 (1 arg)
    and ≥ 5.7 (2 args: step + cumulative).
    """

    def _cb(*maps: "clingo.StatisticsMap") -> None:  # noqa: D401
        stats = maps[-1]  # last positional argument is always a map

        def _get(path: Iterable[str], default=0):
            node = stats
            for key in path:
                node = node.get(key, None)
                if node is None:
                    return default
            return node

        total = _get(["summary", "times", "total"])
        prep = _get(["summary", "times", "preparation"])
        conflicts = _get(["summary", "solver", "conflicts"],
                         _get(["summary", "solving", "conflicts"]))
        choices = _get(["summary", "solver", "choices"],
                       _get(["summary", "solving", "choices"]))
        restarts = _get(["summary", "solver", "restarts"],
                        _get(["summary", "solving", "restarts"]))
        models = _get(["summary", "models"])

        log.info(
            "step %02d | time=%5.3fs (solve=%5.3fs) | "
            "conf=%-6d choices=%-6d restarts=%-4d models=%d",
            step_no, total, total - prep, conflicts, choices, restarts, models,
        )

    return _cb


def model_to_str(model: Sequence[clingo.Symbol]) -> str:
    """Pretty-print a symbol sequence exactly once (no trailing spaces)."""
    return " ".join(f"{a.name}({','.join(map(str, a.arguments))})" for a in model)


# ------------------------------------------------------------------------------
# main driver
# ------------------------------------------------------------------------------


def main() -> None:
    t_start = time.perf_counter()

    # set up external helpers (only used for final pretty print)
    parser = ClingoParser()
    presenter = ClingoPresenter(show_answer_set=False, show_extraction=True)
    template = template_sokoban(*INITIAL_GRID)

    ctl = clingo.Control()
    for f in FILES:
        log.debug("Loading %s", Path(f).name)
        ctl.load(f)

    # ground static base and the first two time steps
    ctl.ground([("base", []), ("step", [Number(1)]), ("step", [Number(2)])])
    log.info("Grounded base and steps 1–2")

    models: list[list[clingo.Symbol]] = []
    ctl.solve(on_model=make_on_model(models), on_statistics=make_on_stats(2))

    if not models:
        log.error("No model for steps 1–2 – aborting.")
        return

    last_model = models.pop()
    assumptions = extract_assumptions(last_model)

    # ------------------------------------------------------------------
    # incremental loop
    # ------------------------------------------------------------------
    for step in range(3, MAX_STEPS + 1):
        ctl.ground([("step", [Number(step)])])

        log.info(
            "=== step %02d | %d assumptions ===",
            step,
            len(assumptions),
        )

        models.clear()
        ctl.solve(
            on_model=make_on_model(models),
            on_statistics=make_on_stats(step),
            assumptions=assumptions,
        )

        if not models:
            log.warning("UNSAT at step %d – stopping.", step)
            break

        last_model = models.pop()
        assumptions = extract_assumptions(last_model)

    # ------------------------------------------------------------------
    # output final model
    # ------------------------------------------------------------------
    final_line = model_to_str(last_model)
    presentation = presenter.present(template, parser.parse_lines([final_line])[0])

    log.info("\nFinal model:\n%s", presentation)
    log.info("Total runtime: %.2fs", time.perf_counter() - t_start)


# ------------------------------------------------------------------------------
# entry-point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
