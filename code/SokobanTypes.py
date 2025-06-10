# sokoban_types.py

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple

Pos = Tuple[int, int]
Trajectory = List[Tuple["State", "Action"]]


class Action(Enum):
    NOOP  = auto()
    LEFT  = auto()
    RIGHT = auto()
    UP    = auto()
    DOWN  = auto()

    def __str__(self):
        return {
            Action.NOOP:  "noop",
            Action.LEFT:  "west",
            Action.RIGHT: "east",
            Action.UP:    "north",
            Action.DOWN:  "south",
        }[self]


@dataclass(order=True, frozen=True)
class Cells:
    bounds: Pos
    walls: List[Pos] = field(default_factory=list)


@dataclass(order=True, frozen=True)
class State:
    cells: Cells
    man: Pos
    blocks: List[Pos] = field(default_factory=list)


@dataclass(order=True, frozen=True)
class Example:
    initial_state: List[str]
    actions: List[Action]
    num_input: int
    num_held_out: int


def empty_strings(bounds: Pos) -> List[str]:
    """Create a grid of '.' of size bounds."""
    bx, by = bounds
    return ["." * bx for _ in range(by)]


def update_strings(grid: List[str], pos: Pos, char: str) -> List[str]:
    """Place `char` at 1-based position `pos` in the grid."""
    x, y = pos
    # Convert to 0-based indices
    row = list(grid[y-1])
    row[x-1] = char
    new_row = "".join(row)
    return grid[:y-1] + [new_row] + grid[y:]


def state_to_strings(state: State) -> List[str]:
    """
    Render the current State as a list of strings:
     - '.' empty
     - 'w' wall
     - 'b' block
     - 'm' man
    """
    # start with empty grid
    grid = empty_strings(state.cells.bounds)

    # draw walls
    for pos in state.cells.walls:
        grid = update_strings(grid, pos, 'w')

    # draw blocks
    for pos in state.blocks:
        grid = update_strings(grid, pos, 'b')

    # draw the man last
    grid = update_strings(grid, state.man, 'm')

    return grid
