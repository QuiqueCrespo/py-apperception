import argparse
import subprocess
import sys
from typing import List

from SokobanTypes import State, Cells, Pos, Trajectory, Action, Example, state_to_strings
from SokobanData2D import sokoban_examples

# ==============================================================================
# Core state manipulation
# ==============================================================================

def empty_state(bounds: Pos) -> State:
    return State(cells=Cells(bounds=bounds, walls=[]), man=(0, 0), blocks=[])


def get_bounds(lines: List[str]) -> Pos:
    return (len(lines[0]), len(lines))


def strings_to_state(lines: List[str]) -> State:
    b = get_bounds(lines)
    state = empty_state(b)
    for y, row in enumerate(lines, start=1):
        for x, ch in enumerate(row, start=1):
            if ch == 'w':
                state = State(
                    cells=Cells(bounds=state.cells.bounds,
                                walls=state.cells.walls + [(x, y)]),
                    man=state.man,
                    blocks=state.blocks
                )
            elif ch == 'b':
                state = State(
                    cells=state.cells,
                    man=state.man,
                    blocks=state.blocks + [(x, y)]
                )
            elif ch == 'm':
                state = State(
                    cells=state.cells,
                    man=(x, y),
                    blocks=state.blocks
                )
    return state


def insert_at(item, index: int, xs: List) -> List:
    return xs[:index] + [item] + xs[index+1:]


def perform_action(state: State, action: Action) -> State:
    if action == Action.NOOP:
        return state
    dx, dy = {
        Action.LEFT: (-1, 0),
        Action.RIGHT: ( 1, 0),
        Action.UP:    ( 0,-1),
        Action.DOWN:  ( 0, 1)
    }[action]
    mx, my = state.man
    new_pos = (mx + dx, my + dy)
    # check if new position is within bounds and possitive
    if new_pos[0] < 1:
        new_pos = (1, new_pos[1])
    if new_pos[1] < 1:
        new_pos = (new_pos[0], 1)
    if new_pos[0] > state.cells.bounds[0]:
        new_pos = (state.cells.bounds[0], new_pos[1])
    if new_pos[1] > state.cells.bounds[1]:
        new_pos = (new_pos[0], state.cells.bounds[1])
    # check if new position is a wall

    blocks = state.blocks
    # push block if present
    if new_pos in blocks:
        beyond = (mx + 2*dx, my + 2*dy)
        if beyond in blocks:
            raise ValueError("Invalid push: no empty space")
        blocks = insert_at(beyond, blocks.index(new_pos), blocks)
    return State(cells=state.cells, man=new_pos, blocks=blocks)

# ==============================================================================
# Trajectory functions
# ==============================================================================

def example_to_trajectory(ex: Example) -> Trajectory:
    traj: Trajectory = [(strings_to_state(ex.initial_state), ex.actions[0])]
    i = 1
    while i <= len(ex.actions):
       
        prev_state, prev_action = traj[-1]
        new_state = perform_action(prev_state, prev_action)
        if i == len(ex.actions):
            traj.append((new_state, Action.NOOP))
            break
        traj.append((new_state, ex.actions[i]))
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
    for x in range(1, bx+1):
        for y in range(1, by+1):
            tag = 'p_is_wall' if (x, y) in walls else 'p_is_not_wall'
            lines.append(f"permanent(isa({tag}, obj_cell_{x}_{y})).")
    return lines


def cell_adjacency(traj: Trajectory) -> List[str]:
    bx, by = traj[0][0].cells.bounds
    lines = ["% Cell adjacency"]
    # rights
    for y in range(1, by+1):
        for x in range(1, bx):
            lines.append(
                f"permanent(isa2(p_right, obj_cell_{x}_{y}, obj_cell_{x+1}_{y}))."
            )
    # belows
    for x in range(1, bx+1):
        for y in range(1, by):
            lines.append(
                f"permanent(isa2(p_below, obj_cell_{x}_{y}, obj_cell_{x}_{y+1}))."
            )
    lines.append("")
    return lines


def times(traj: Trajectory) -> List[str]:
    length = len(traj)
    return ["% Time", f"is_time(1..{length}).", ""]


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
        ""
    ]


def exists_uniques() -> List[str]:
    lines: List[str] = []
    lines += exists_unique("c_in_1", "t_1")
    lines += exists_unique("c_in_2", "t_2")
    return lines


def xors() -> List[str]:
    actions = ["c_noop", "c_east", "c_west", "c_north", "c_south"]
    pairs = [(a1, a2) for i, a1 in enumerate(actions) for a2 in actions[i+1:]]
    lines = ["% Exclusions", "% Every action is either noop, north, south, east, or west", "% ∀X : man, noop(X) ⊕ north(X) ⊕ south(X) ⊕ east(X) ⊕ west(X)", "", "% At most one"]
    for a1, a2 in pairs:
        lines.extend([
            ":-",
            f"\tholds(s({a1}, X), t),",
            f"\tholds(s({a2}, X), t)."
        ])
    lines.append("")
    lines.append("% At least one")
    lines.extend([
        ":-",
        "\tpermanent(isa(t_1, X)),",
        "\tis_time(t),",
        "\tnot holds(s(c_noop, X), t),",
        "\tnot holds(s(c_east, X), t),",
        "\tnot holds(s(c_west, X), t),",
        "\tnot holds(s(c_north, X), t),",
        "\tnot holds(s(c_south, X), t).",
        "",
        "#program base.",
        "% Incompossibility"
    ])
    for a1, a2 in pairs:
        lines.extend([
            f"incompossible(s({a1}, X), s({a2}, X)) :-",
            "\tpermanent(isa(t_1, X)).",
            ""
        ])
    return lines


def concepts() -> List[str]:
    items = ["in_1", "in_2", "noop", "east", "west", "north", "south"]
    lines = ["% Concepts"]
    for s in items:
        lines.append(f"is_concept({s}).")
    lines.append("")
    return lines

def actions() -> List[str]:
    lines = ["% Actions"]
    for s in ["c_noop", "c_east", "c_west", "c_north", "c_south"]:
        lines.append(f"is_exogenous({s}).")
    lines.append("")
    return lines


def comments(traj: Trajectory) -> List[str]:
    lines: List[str] = []
    header = ["%--------------------------------------------------", "% Generated by main.py", "%--------------------------------------------------", "% "]
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
        lines.append(f"exogenous(s(c_{a}, obj_x1), {i}).")
    lines.append("")
    return lines


def trajectory_atoms(traj: Trajectory) -> List[str]:
    lines = ["% The given sequence"]
    for i, (s, _) in enumerate(traj, start=1):
        mx, my = s.man
        lines.append(
            f"senses(s2(c_in_1, obj_x1, obj_cell_{mx}_{my}), {i})."
        )
        for j, (bx, by) in enumerate(s.blocks, start=2):
            lines.append(
                f"senses(s2(c_in_2, obj_x{j}, obj_cell_{bx}_{by}), {i})."
            )
    lines.append("")
    return lines


def elements(traj: Trajectory) -> List[str]:
    state, _ = traj[0]
    bx, by = state.cells.bounds
    lines: List[str] = ["% Elements"]
    lines.append("is_object(obj_x1).")
    for idx in range(2, len(state.blocks)+2):
        lines.append(f"is_object(obj_x{idx}).")
    for x in range(1, bx+1):
        for y in range(1, by+1):
            lines.append(f"is_object(obj_cell_{x}_{y}).")
    for x in range(1, bx+1):
        for y in range(1, by+1):
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
    # output += times(traj)
    output += ["#program step(t)."]
    output += exists_uniques()
    output += xors()
    
    return output


def gen_symbolic_sokoban(name: str, ex: Example, out_dir: str = 'data/sokoban'):
    print(f"Processing example {name}")
    content = '\n'.join(example_to_symbolic_strings(ex)) + '\n'
    path = f"{out_dir}/predict_{name}.lp"
    with open(path, 'w') as f:
        f.write(content)


def gen_symbolic_all():
    for name, ex in sokoban_examples:
        gen_symbolic_sokoban(name, ex)
    # generate single experiment script
    script_lines = ['#!/bin/bash', '', 'case $(expr $1 + 1) in']
    for idx, (name, _) in enumerate(sokoban_examples, start=1):
        script_lines.append(f"    {idx} )")
        script_lines.append(f"        echo \"Solving sokoban example {name}...\"")
        script_lines.append(f"        time code/solve sokoban {name}")
        script_lines.append("        ;;")
    script_lines.append('esac')
    script = '\n'.join(script_lines) + '\n'
    scripts_path = 'scripts/single_sokoban.sh'
    print(f"Generating file {scripts_path}")
    with open(scripts_path, 'w') as f:
        f.write(script)
    subprocess.call(['chmod', '777', scripts_path])


def parse_args():
    parser = argparse.ArgumentParser(description='Sokoban symbolic generator')
    parser.add_argument('example', nargs='?', help='Example name or "all"')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.example or args.example == 'all':
        gen_symbolic_all()
    else:
        lookup = dict(sokoban_examples)
        if args.example in lookup:
            gen_symbolic_sokoban(args.example, lookup[args.example])
        else:
            sys.exit(f"No example called {args.example}")


if __name__ == '__main__':
    main()
