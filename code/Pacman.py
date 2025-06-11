import argparse
import os
import subprocess
import sys
from typing import List

from PacmanTypes import State, Cells, Pos, Trajectory, PacmanAction, state_to_strings
from PacmanData2D import pacman_examples, PacmanAction as DataAction, Example

# ==============================================================================
# Core state manipulation
# ==============================================================================

def empty_state(bounds: Pos) -> State:
    return State(
        cells=Cells(bounds=bounds, walls=[]),
        pacman=(0, 0),
        pellets=[],
        ghosts=[],
        ghost_dirs=[],
        powered=False,
        alive=True,
        pacman_alive=True,
        ghost_alive=[],
    )


def get_bounds(lines: List[str]) -> Pos:
    return (len(lines[0]), len(lines))


def strings_to_state(lines: List[str]) -> State:
    b = get_bounds(lines)
    walls: List[Pos] = []
    pellets: List[Pos] = []
    ghosts: List[Pos] = []
    ghost_dirs: List[PacmanAction] = []
    ghost_alive: List[bool] = []
    pacman: Pos = (0, 0)

    for y, row in enumerate(lines, start=1):
        for x, ch in enumerate(row, start=1):
            if ch == '#':
                walls.append((x, y))
            elif ch == 'o':
                pellets.append((x, y))
            elif ch == 'g':
                ghosts.append((x, y))
                ghost_dirs.append(PacmanAction.LEFT)
                ghost_alive.append(True)
            elif ch == 'p':
                pacman = (x, y)

    cells = Cells(bounds=b, walls=walls)
    return State(
        cells=cells,
        pacman=pacman,
        pellets=pellets,
        ghosts=ghosts,
        ghost_dirs=ghost_dirs,
        ghost_alive=ghost_alive,
        powered=False,
        alive=True,
        pacman_alive=True,
    )


def convert_action(a: DataAction) -> PacmanAction:
    return {
        DataAction.STOP: PacmanAction.NOOP,
        DataAction.LEFT: PacmanAction.LEFT,
        DataAction.RIGHT: PacmanAction.RIGHT,
        DataAction.UP: PacmanAction.UP,
        DataAction.DOWN: PacmanAction.DOWN,
    }[a]


def perform_action(state: State, action: PacmanAction) -> State:
    """Advance the Pacman state by one action."""

    # mapping from action to delta movement
    delta = {
        PacmanAction.LEFT: (-1, 0),
        PacmanAction.RIGHT: (1, 0),
        PacmanAction.UP: (0, -1),
        PacmanAction.DOWN: (0, 1),
    }

    opposite = {
        PacmanAction.LEFT: PacmanAction.RIGHT,
        PacmanAction.RIGHT: PacmanAction.LEFT,
        PacmanAction.UP: PacmanAction.DOWN,
        PacmanAction.DOWN: PacmanAction.UP,
    }

    # ------------------------------------------------------------------
    # move pacman
    # ------------------------------------------------------------------
    px, py = state.pacman
    if action != PacmanAction.NOOP:
        dx, dy = delta[action]
        new_px, new_py = px + dx, py + dy
    else:
        new_px, new_py = px, py

    bx, by = state.cells.bounds
    new_px = max(1, min(bx, new_px))
    new_py = max(1, min(by, new_py))

    if (new_px, new_py) in state.cells.walls:
        new_px, new_py = px, py

    pacman_pos = (new_px, new_py)

    # pellet pickup and power state
    powered = state.powered or (pacman_pos in state.pellets)
    pellets = [p for p in state.pellets if p != pacman_pos]

    # ------------------------------------------------------------------
    # move ghosts
    # ------------------------------------------------------------------
    ghosts: List[Pos] = []
    ghost_dirs: List[PacmanAction] = []
    alive = state.alive

    for pos, direction in zip(state.ghosts, state.ghost_dirs):
        gx, gy = pos
        dx, dy = delta[direction]
        next_pos = (gx + dx, gy + dy)
        if next_pos in state.cells.walls:
            direction = opposite[direction]
            dx, dy = delta[direction]
            next_pos = (gx + dx, gy + dy)

        # handle collision with pacman
        if next_pos == pacman_pos:
            if powered:
                # ghost eaten: do not append to new lists
                continue
            else:
                alive = False

        ghosts.append(next_pos)
        ghost_dirs.append(direction)

    return State(
        cells=state.cells,
        pacman=pacman_pos,
        pellets=pellets,
        ghosts=ghosts,
        ghost_dirs=ghost_dirs,
        powered=powered,
        alive=alive,
    )


# ==============================================================================
# Trajectory functions
# ==============================================================================

def example_to_trajectory(ex: Example) -> Trajectory:
    actions = [convert_action(a) for a in ex.actions]
    traj: Trajectory = [(strings_to_state(ex.initial_state), actions[0])]
    i = 1
    while i <= len(actions):
        prev_state, prev_action = traj[-1]
        result = perform_action(prev_state, prev_action)

        # ``perform_action`` may return either just a State or a tuple
        # ``(State, alive)``.  Handle both cases gracefully.
        if isinstance(result, tuple) and len(result) == 2:
            new_state, alive = result
        else:  # Backwards compatibility
            new_state, alive = result, getattr(result, "alive", True)

        if i == len(actions) or not alive:
            # Always append the final state paired with ``NOOP``
            traj.append((new_state, PacmanAction.NOOP))
            break

        traj.append((new_state, actions[i]))
        i += 1

    return traj


def trajectory_to_strings(traj: Trajectory) -> List[str]:
    lines: List[str] = []
    for state, action in traj:
        lines.extend(state_to_strings(state))
        lines.append("")
        lines.append(str(action))
        lines.append("")
    return lines


def print_example(ex: Example):
    traj = example_to_trajectory(ex)
    for line in trajectory_to_strings(traj):
        print(line)
    print(f"Length: {len(traj)}")

# ==============================================================================
# Symbolic output generators
# ==============================================================================

def p_walls(traj: Trajectory) -> List[str]:
    state, _ = traj[0]
    bx, by = state.cells.bounds
    walls = state.cells.walls
    lines = ["% Walls"]
    for x in range(1, bx + 1):
        for y in range(1, by + 1):
            tag = 'p_is_wall' if (x, y) in walls else 'p_is_not_wall'
            lines.append(f"permanent(isa({tag}, obj_cell_{x}_{y})).")
    return lines


def cell_adjacency(traj: Trajectory) -> List[str]:
    bx, by = traj[0][0].cells.bounds
    lines = ["% Cell adjacency"]
    for y in range(1, by + 1):
        for x in range(1, bx):
            lines.append(f"permanent(isa2(p_right, obj_cell_{x}_{y}, obj_cell_{x+1}_{y})).")
    for x in range(1, bx + 1):
        for y in range(1, by):
            lines.append(f"permanent(isa2(p_below, obj_cell_{x}_{y}, obj_cell_{x}_{y+1})).")
    lines.append("")
    return lines


def exists_unique(p: str, t: str) -> List[str]:
    return [
        f"% ∃! clause for {p} : at most one",
        ":-",
        f"\tholds(s2({p}, X, Y), t),",
        f"\tholds(s2({p}, X, Y2), t),",
        "\tY != Y2.",
        "",
        f"% ∃! clause for {p} : at least one",
        ":-",
        f"\tpermanent(isa({t}, X)),",
        "\tis_time(t),",
        f"\tnot aux_{p}(X, t).",
        "",
        f"aux_{p}(X, t) :-",
        f"\tholds(s2({p}, X, _), t).",
        "",
        f"% Incompossibility for {p}",
        f"incompossible(s2({p}, X, Y), s2({p}, X, Y2)) :-",
        f"\tpermanent(isa({t}, X)),",
        "\tpermanent(isa(t_cell, Y)),",
        "\tpermanent(isa(t_cell, Y2)),",
        "\tY != Y2.",
        "",
    ]


def exists_uniques() -> List[str]:
    lines: List[str] = []
    lines += exists_unique("c_pacman_at", "t_pacman")
    lines += exists_unique("c_pellet_at", "t_pellet")
    lines += exists_unique("c_ghost_at", "t_ghost")
    return lines


def xors() -> List[str]:
    actions = ["c_noop", "c_left", "c_right", "c_up", "c_down"]
    pairs = [(a1, a2) for i, a1 in enumerate(actions) for a2 in actions[i + 1:]]
    lines = [
        "% Exclusions",
        "% Every action is either noop, up, down, left, or right",
        "% ∀X : pacman, noop(X) ⊕ up(X) ⊕ down(X) ⊕ left(X) ⊕ right(X)",
        "",
        "% At most one",
    ]
    for a1, a2 in pairs:
        lines.extend([
            ":-",
            f"\tholds(s({a1}, X), t),",
            f"\tholds(s({a2}, X), t).",
        ])
    lines.append("")
    lines.append("% At least one")
    lines.extend([
        ":-",
        "\tpermanent(isa(t_pacman, X)),",
        "\tis_time(t),",
        "\tnot holds(s(c_noop, X), t),",
        "\tnot holds(s(c_left, X), t),",
        "\tnot holds(s(c_right, X), t),",
        "\tnot holds(s(c_up, X), t),",
        "\tnot holds(s(c_down, X), t).",
        "",
        "#program base.",
        "% Incompossibility",
    ])
    for a1, a2 in pairs:
        lines.extend([
            f"incompossible(s({a1}, X), s({a2}, X)) :-",
            "\tpermanent(isa(t_pacman, X)).",
            "",
        ])
    return lines


def concepts() -> List[str]:
    items = ["pacman_at", "pellet_at", "ghost_at", "noop", "left", "right", "up", "down"]
    lines = ["% Concepts"]
    for s in items:
        lines.append(f"is_concept({s}).")
    lines.append("")
    return lines


def actions() -> List[str]:
    lines = ["% Actions"]
    for s in ["c_noop", "c_left", "c_right", "c_up", "c_down"]:
        lines.append(f"is_exogenous({s}).")
    lines.append("")
    return lines


def comments(traj: Trajectory) -> List[str]:
    lines: List[str] = []
    header = [
        "%--------------------------------------------------",
        "% Generated by Pacman.py",
        "%--------------------------------------------------",
        "% ",
    ]
    lines += header
    for i, (s, a) in enumerate(traj, start=1):
        lines.append(f"% Time {i}:")
        for l in state_to_strings(s):
            lines.append(f"% {l}")
        lines.append(f"% {a}")
        lines.append("% ")
    lines.append("")
    return lines


def exogenous_atoms(traj: Trajectory) -> List[str]:
    lines = ["% Exogenous actions"]
    for i, (_, a) in enumerate(traj, start=1):
        lines.append(f"exogenous(s(c_{a}, obj_pacman), {i}).")
    lines.append("")
    return lines


def trajectory_atoms(traj: Trajectory) -> List[str]:
    lines = ["% The given sequence"]
    for i, (s, _) in enumerate(traj, start=1):
        px, py = s.pacman
        lines.append(f"senses(s2(c_pacman_at, obj_pacman, obj_cell_{px}_{py}), {i}).")
        for j, (ox, oy) in enumerate(s.pellets, start=1):
            lines.append(f"senses(s2(c_pellet_at, obj_pellet_{j}, obj_cell_{ox}_{oy}), {i}).")
        for j, (gx, gy) in enumerate(s.ghosts, start=1):
            lines.append(f"senses(s2(c_ghost_at, obj_ghost_{j}, obj_cell_{gx}_{gy}), {i}).")
    lines.append("")
    return lines


def elements(traj: Trajectory) -> List[str]:
    state, _ = traj[0]
    bx, by = state.cells.bounds
    lines: List[str] = ["% Elements"]
    lines.append("is_object(obj_pacman).")
    for idx in range(1, len(state.pellets) + 1):
        lines.append(f"is_object(obj_pellet_{idx}).")
    for idx in range(1, len(state.ghosts) + 1):
        lines.append(f"is_object(obj_ghost_{idx}).")
    for x in range(1, bx + 1):
        for y in range(1, by + 1):
            lines.append(f"is_object(obj_cell_{x}_{y}).")
    for x in range(1, bx + 1):
        for y in range(1, by + 1):
            lines.append(f"is_cell(obj_cell_{x}_{y}).")
    lines.append("")
    return lines

# ==============================================================================
# Main and file generation
# ==============================================================================

def example_to_symbolic_strings(ex: Example) -> List[str]:
    traj = example_to_trajectory(ex)
    output: List[str] = ["#program base."]
    output += comments(traj)
    output += trajectory_atoms(traj)
    output += exogenous_atoms(traj)
    output += elements(traj)
    output += concepts()
    output += actions()
    output += cell_adjacency(traj)
    output += p_walls(traj)
    output += ["#program step(t)."]
    output += exists_uniques()
    output += xors()
    return output


def gen_symbolic_pacman(name: str, ex: Example, out_dir: str = 'data/pacman'):
    print(f"Processing example {name}")
    content = '\n'.join(example_to_symbolic_strings(ex)) + '\n'
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/predict_{name}.lp"
    with open(path, 'w') as f:
        f.write(content)


def gen_symbolic_all():
    for name, ex in pacman_examples:
        gen_symbolic_pacman(name, ex)


def parse_args():
    parser = argparse.ArgumentParser(description='Pacman symbolic generator')
    parser.add_argument('example', nargs='?', help='Example name or "all"')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.example or args.example == 'all':
        gen_symbolic_all()
    else:
        lookup = dict(pacman_examples)
        if args.example in lookup:
            gen_symbolic_pacman(args.example, lookup[args.example])
        else:
            sys.exit(f"No example called {args.example}")


if __name__ == '__main__':
    main()
