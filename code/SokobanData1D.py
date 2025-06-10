from typing import List, Tuple

from SokobanTypes import Example, Action

# 1D Sokoban examples

state_1: List[str] = ["..m.."]
example_1: Example = Example(
    initial_state=state_1,
    actions=[
        Action.RIGHT, Action.RIGHT,
        Action.LEFT, Action.LEFT,
        Action.RIGHT, Action.LEFT, Action.RIGHT
    ],
    num_input=0,
    num_held_out=0
)

state_2: List[str] = ["..m.b..."]
example_2: Example = Example(
    initial_state=state_2,
    actions=[
        Action.RIGHT, Action.RIGHT, Action.RIGHT,
        Action.LEFT, Action.RIGHT, Action.LEFT, Action.LEFT
    ],
    num_input=0,
    num_held_out=0
)

state_3: List[str] = [".b.m.b.."]
example_3: Example = Example(
    initial_state=state_3,
    actions=[
        Action.RIGHT, Action.RIGHT,
        Action.LEFT, Action.LEFT, Action.LEFT, Action.LEFT,
        Action.RIGHT
    ],
    num_input=0,
    num_held_out=0
)

# Helper functions to generate variations

def make_example(ex: Example, n: int) -> Example:
    """
    Produce a new Example where the last n actions are held out.
    """
    total = len(ex.actions)
    kept = ex.actions[: total - n] if n <= total else []
    return Example(
        initial_state=ex.initial_state,
        actions=kept,
        num_input=total + 1 - n,
        num_held_out=n
    )


def make_n_examples(n: int, ex: Example) -> List[Tuple[str, Example]]:
    """
    For a given example, produce 7 variants named e1d_{n}_{i} for i in 0..6.
    """
    variants: List[Tuple[str, Example]] = []
    for i in range(7):
        name = f"e1d_{n}_{i}"
        variants.append((name, make_example(ex, i)))
    return variants


# Base list of examples
examples: List[Example] = [example_1, example_2, example_3]

# Flattened list of named example variants
sokoban_examples: List[Tuple[str, Example]] = []
for idx, ex in enumerate(examples):
    sokoban_examples.extend(make_n_examples(idx, ex))

# Optionally, expose module-level variables
__all__ = [
    "examples",
    "make_example",
    "make_n_examples",
    "sokoban_examples"
]
