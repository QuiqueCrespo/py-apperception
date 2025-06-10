from typing import List, Tuple

from SokobanTypes import Example, Action

# 2D Sokoban examples with longer action sequences

state_2d_1: List[str] = [
    ".....",
    "..m..",
    ".b...",
    "....."
]
example_2d_1: Example = Example(
    initial_state=state_2d_1,
    actions=[
        Action.DOWN, Action.LEFT,
        Action.UP, Action.RIGHT,
        Action.DOWN, Action.RIGHT,
        Action.UP, Action.LEFT,
        Action.DOWN, Action.RIGHT,
        Action.UP, Action.RIGHT
    ],  # 12 moves
    num_input=0,
    num_held_out=0
)

state_2d_2: List[str] = [
    ".......",
    ".m.b...",
    "...b..",
    ".......",
    ".......",
]
example_2d_2: Example = Example(
    initial_state=state_2d_2,
    actions=[
        Action.RIGHT, Action.RIGHT,
        Action.DOWN,  Action.RIGHT,
        Action.UP, Action.LEFT, Action.LEFT,
        Action.DOWN, Action.DOWN,
        Action.LEFT, Action.LEFT,
        Action.LEFT, Action.UP,
        

    ],  # 12 moves
    num_input=0,
    num_held_out=0
)

state_2d_3: List[str] = [
    "......",
    ".b.m.b",
    "......",
    "......",
]
example_2d_3: Example = Example(
    initial_state=state_2d_3,
    actions=[
        Action.LEFT, Action.LEFT,
        Action.RIGHT, Action.DOWN,
        Action.UP, Action.RIGHT,
        Action.LEFT, Action.UP,
        Action.DOWN, Action.LEFT,
        Action.RIGHT, Action.DOWN
    ],  # 12 moves
    num_input=0,
    num_held_out=0
)

# Helper functions (same as 1D)

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
    For a given example, produce 7 variants named e2d_{n}_{i} for i in 0..6.
    """
    variants: List[Tuple[str, Example]] = []
    for i in range(1):
        name = f"e2d_{n}_{i}"
        variants.append((name, make_example(ex, i)))
    return variants

# Base list of 2D examples
examples_2d: List[Example] = [example_2d_1, example_2d_2, example_2d_3]

# Flattened list of named 2D example variants
sokoban_examples: List[Tuple[str, Example]] = []
for idx, ex in enumerate(examples_2d):
    sokoban_examples.extend(make_n_examples(idx, ex))

# Optionally, expose module-level variables
__all__ = [
    "examples_2d",
    "make_example",
    "make_n_examples",
    "sokoban_2d_examples"
]
