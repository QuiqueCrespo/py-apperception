from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

from PacmanTypes import PacmanAction
from PacmanTypes import Example

# ---------------------------------------------------------------------------
# Example task descriptions
# ---------------------------------------------------------------------------

state_2d_1: List[str] = [
    "#######",
    "#.....#",
    "#.p.o.#",
    "#..g..#",
    "#######",
]
example_2d_1 = Example(
    initial_state=state_2d_1,
    actions=[
        PacmanAction.DOWN, PacmanAction.RIGHT,
        PacmanAction.UP, PacmanAction.LEFT,
        PacmanAction.DOWN, PacmanAction.RIGHT,
        PacmanAction.UP, PacmanAction.LEFT,
        PacmanAction.DOWN, PacmanAction.RIGHT,
        PacmanAction.UP, PacmanAction.LEFT,
    ],
    num_input=0,
    num_held_out=0,
)

state_2d_2: List[str] = [
    "#########",
    "#p..o...#",
    "###.###.#",
    "#..g....#",
    "#########",
]
example_2d_2 = Example(
    initial_state=state_2d_2,
    actions=[
        PacmanAction.RIGHT, PacmanAction.RIGHT,
        PacmanAction.DOWN, PacmanAction.LEFT,
        PacmanAction.UP, PacmanAction.UP,
        PacmanAction.RIGHT, PacmanAction.DOWN,
        PacmanAction.LEFT, PacmanAction.LEFT,
        PacmanAction.UP, PacmanAction.RIGHT,
    ],
    num_input=0,
    num_held_out=0,
)

state_2d_3: List[str] = [
    "########",
    "#g..p..#",
    "#..o#..#",
    "#......#",
    "########",
]
example_2d_3 = Example(
    initial_state=state_2d_3,
    actions=[
        PacmanAction.LEFT, PacmanAction.LEFT,
        PacmanAction.RIGHT, PacmanAction.UP,
        PacmanAction.DOWN, PacmanAction.RIGHT,
        PacmanAction.LEFT, PacmanAction.DOWN,
        PacmanAction.UP, PacmanAction.RIGHT,
        PacmanAction.NOOP, PacmanAction.LEFT,
    ],
    num_input=0,
    num_held_out=0,
)


# ---------------------------------------------------------------------------
# Variant generation utilities
# ---------------------------------------------------------------------------


def make_example(ex: Example, n: int) -> Example:
    """Return a copy of ``ex`` with the last ``n`` actions held out."""
    total = len(ex.actions)
    kept = ex.actions[: total - n] if n <= total else []
    return Example(
        initial_state=ex.initial_state,
        actions=kept,
        num_input=total + 1 - n,
        num_held_out=n,
    )


def make_n_examples(n: int, ex: Example) -> List[Tuple[str, Example]]:
    """Generate seven variants of an example named ``e2d_{n}_{i}``."""
    variants: List[Tuple[str, Example]] = []
    for i in range(1):
        name = f"e2d_{n}_{i}"
        variants.append((name, make_example(ex, i)))
    return variants


# Base examples
examples_2d: List[Example] = [example_2d_1, example_2d_2, example_2d_3]

# Flatten all named example variants
pacman_examples: List[Tuple[str, Example]] = []
for idx, ex in enumerate(examples_2d):
    pacman_examples.extend(make_n_examples(idx, ex))

__all__ = [
    "examples_2d",
    "make_example",
    "make_n_examples",
    "pacman_examples",
    "PacmanAction",
    "Example",
]
