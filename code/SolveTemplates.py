# Interpretation module provides core types and constructors
from Interpretation import (
    Frame,
    Template,
    Type as T,
    Object as O,
    Var as V,
    C,
    P,
    ConceptLineage,
)





def frame_sokoban(max_x: int, max_y: int, num_blocks: int) -> Frame:
    return Frame(
        types=[T("cell"), T("1"), T("2")],
        type_hierarchy=[],
        objects=[(O("x1"), T("1"))]
                + [(O(f"cell_{i}_{j}"), T("cell")) for i in range(1, max_x+1) for j in range(1, max_y+1)]
                + [(O(f"x{i+1}"), T("2")) for i in range(1, num_blocks+1)],
        exogeneous_objects=[],
        permanent_concepts=[
            (P("right"), ConceptLineage.GIVEN, [T("cell"), T("cell")]),
            (P("below"), ConceptLineage.GIVEN, [T("cell"), T("cell")]),
            (P("is_not_wall"), ConceptLineage.GIVEN, [T("cell")]),
            (P("is_wall"), ConceptLineage.GIVEN, [T("cell")]),
        ],
        fluid_concepts=[
            (C("in_1"), [T("1"), T("cell")]),
            (C("in_2"), [T("2"), T("cell")]),
            (C("noop"), [T("1")]),
            (C("east"), [T("1")]),
            (C("west"), [T("1")]),
            (C("north"), [T("1")]),
            (C("south"), [T("1")]),
            (C("p1"), [T("2")]),
            (C("p2"), [T("2")]),
            (C("p3"), [T("2")]),
            (C("p4"), [T("2")]),
        ],
        input_concepts=[C("in_1"), C("in_2"), C("noop"), C("east"), C("west"), C("north"), C("south"), C("p1"), C("p2"), C("p3"), C("p4")],
        static_concepts=[C("p1"), C("p2"), C("p3"), C("p4")],
        vars=[
            (V("c1"), T("cell")),
            (V("c2"), T("cell")),
            (V("x"), T("1")),
            (V("y"), T("2"))
        ],
        var_groups=[
            [V("c1"), V("c2"), V("x")],
            [V("c1"), V("c2"), V("x"), V("y")],

        ],
        aux_files=[]
    )


def template_sokoban(max_x: int, max_y: int, num_blocks: int) -> Template:
    return Template(
        dir="sokoban",
        frame=frame_sokoban(max_x, max_y, num_blocks),
        min_body_atoms=1,
        max_body_atoms=4,
        num_arrow_rules=4,
        num_causes_rules=8,
        use_noise=False,
        num_visual_predicates=None
        
    )

# Iterative-solving and template utilities omitted for brevity (see above)


def make_eca_template(hard_code_space, input_f, n):
    """
    Creates an ECA template with a specific number of causes rules.

    Args:
        hard_code_space (bool): Whether to include 'space11.lp' as an auxiliary file.
        input_f (str): A descriptive string (unused in current Haskell logic but kept for signature).
        n (int): The number of 'causes' rules to set in the template.

    Returns:
        tuple: A tuple containing a descriptive string and the configured Template object.
    """
    # Haskell: s = "Using " ++ show n ++ " causes rules"
    s = f"Using {n} causes rules"
    
    # Haskell: t = (template_eca hard_code_space) { num_causes_rules = n }
    # In Python, we create a copy of the base template and then modify it.
    base_template = template_eca(hard_code_space)
    base_template.num_causes_rules = n
    t = base_template
    
    return (s, t)

def frame_eca(hard_code_space, n):
    """
    Defines the frame for an ECA, specifying types, objects, concepts, and variables.

    Args:
        hard_code_space (bool): If True, a permanent concept 'r' is 'Given' and 'space11.lp' is included.
                                If False, 'r' is 'Constructed'.
        n (int): The number of 'sensor' objects (cell_1 to cell_n).

    Returns:
        Frame: The configured Frame object for ECA.
    """
    return Frame(
        types=[T("sensor")],
        type_hierarchy=[],
        objects=[(O(f"cell_{i}"), T("sensor")) for i in range(1, n + 1)],
        exogeneous_objects=[],
        permanent_concepts=[
            (
                P("r"),
                ConceptLineage.GIVEN if hard_code_space else ConceptLineage.CONSTRUCTED,
                [T("sensor"), T("sensor")],
            )
        ],
        fluid_concepts=[
            (C("on"), [T("sensor")]),
            (C("off"), [T("sensor")]),
        ],
        input_concepts=[C("on"), C("off")],
        static_concepts=[],
        vars=[
            (V("s"), T("sensor")),
            (V("s2"), T("sensor")),
            (V("s3"), T("sensor")),
        ],
        var_groups=[
            [V("s"), V("s2"), V("s3")],
        ],
        aux_files=["aux_eca_space11.lp"] if hard_code_space else []
    )

def template_eca_n(hard_code_space, n_sensors):
    """
    Creates a basic ECA template with a configurable number of sensors.

    Args:
        hard_code_space (bool): Passed to frame_eca.
        n_sensors (int): The number of sensor objects in the frame.

    Returns:
        Template: The configured Template object.
    """
    return Template(
        dir="eca",
        frame=frame_eca(hard_code_space, n_sensors),
        min_body_atoms=1,
        max_body_atoms=5,
        num_arrow_rules=0,
        num_causes_rules=4, # Default value for num_causes_rules
        num_visual_predicates=None, # Equivalent to Nothing in Haskell
        use_noise=False
    )

def template_eca(hard_code_space):
    """
    The default ECA template, using 11 sensors.

    Args:
        hard_code_space (bool): Passed to template_eca_n.

    Returns:
        Template: The default ECA Template object.
    """
    return template_eca_n(hard_code_space, 11)

# # A minimal frame for ECA with just two sensors
# frame_eca_small = Frame(
#     types=[T("sensor"), T("grid"), T("object")],
#     type_hierarchy=[
#         (T("object"), [T("sensor"), T("grid")])
#     ],
#     objects=[
#         (O("cell_1"), T("sensor")),
#         (O("cell_2"), T("sensor")),
#         (O("grid"), T("grid"))
#     ],
#     exogeneous_objects=[],
#     permanent_concepts=[],
#     fluid_concepts=[
#         (C("on"), [T("sensor")]),
#         (C("off"), [T("sensor")]),
#         (C("part"), [T("sensor"), T("grid")]),
#     ],
#     input_concepts=[C("on"), C("off")],
#     static_concepts=[],
#     vars=[
#         (V("s"), T("sensor")),
#         (V("s2"), T("sensor")),
#     ],
#     var_groups=[
#         [V("s")],
#         [V("s"), V("s2")],
#     ],
#     aux_files=[]
# )

# template_eca_small = Template(
#     dir="eca",
#     frame=frame_eca_small,
#     min_body_atoms=0,
#     max_body_atoms=2,
#     num_arrow_rules=1,
#     num_causes_rules=3,
#     num_visual_predicates=None,
#     use_noise=False
# )


def frame_pacman(max_x: int, max_y: int, num_pellets: int, num_ghosts: int) -> Frame:
    """Return a basic Pacman frame."""

    return Frame(
        types=[T("cell"), T("character"), T("pacman"), T("pellet"), T("ghost")],
        type_hierarchy=[(T("character"), [T("pacman"), T("ghost")])],
        objects=[(O("pacman"), T("pacman"))]
                + [(O(f"pellet_{i}"), T("pellet")) for i in range(1, num_pellets + 1)]
                + [(O(f"ghost_{i}"), T("ghost")) for i in range(1, num_ghosts + 1)]
                + [
                    (O(f"cell_{x}_{y}"), T("cell"))
                    for x in range(1, max_x + 1)
                    for y in range(1, max_y + 1)
                ],
        exogeneous_objects=[],
        permanent_concepts=[
            (P("right"), ConceptLineage.GIVEN, [T("cell"), T("cell")]),
            (P("below"), ConceptLineage.GIVEN, [T("cell"), T("cell")]),
            (P("is_not_wall"), ConceptLineage.GIVEN, [T("cell")]),
            (P("is_wall"), ConceptLineage.GIVEN, [T("cell")]),
        ],
        fluid_concepts=[
            (C("pacman_at"), [T("pacman"), T("cell")]),
            (C("ghost_at"), [T("ghost"), T("cell")]),
            (C("pellet_at"), [T("pellet"), T("cell")]),
            (C("alive"), [T("character")]),
            (C("dead"), [T("character")]),
            (C("noop"), [T("pacman")]),
            (C("west"), [T("pacman")]),
            (C("east"), [T("pacman")]),
            (C("north"), [T("pacman")]),
            (C("south"), [T("pacman")]),

            (C("p1"), [T("ghost")]),
            (C("p2"), [T("ghost")]),
            (C("p3"), [T("pacman")]),
            # (C("p4"), [T("object")]),
        ],
        input_concepts=[
            C("pacman_at"),
            C("ghost_at"),
            C("pellet_at"),
            C("alive"),
            C("dead"),
            C("noop"),
            C("west"),
            C("east"),
            C("north"),
            C("south"),
            C("p1"),
            C("p2"),
            C("p3"),
            # C("p4")
        ],
        static_concepts=[],
        vars=[
            (V("c1"), T("cell")),
            (V("c2"), T("cell")),
            (V("p"), T("pacman")),
            (V("g"), T("ghost")),
            (V("o"), T("pellet")),
        ],
        var_groups=[
            [V("p"), V("c1")],
            [V("g"), V("c1")],
            [V("o"), V("c1")],
            [V("p"), V("c1"), V("c2")],
            [V("g"), V("c1"), V("c2")],
            [V("o"), V("c1"), V("c2")],
            [V("p"), V("c1"), V("c2"), V("g")],
            [V("o"), V("c1"), V("c2"), V("p")],
        
        ],
        aux_files=["aux_pacman.lp"],
    )


def template_pacman(max_x: int, max_y: int, num_pellets: int, num_ghosts: int) -> Template:
    """Template for solving Pacman trajectories."""

    return Template(
        dir="pacman",
        frame=frame_pacman(max_x, max_y, num_pellets, num_ghosts),
        min_body_atoms=1,
        max_body_atoms=4,
        num_arrow_rules=0,
        num_causes_rules=7,
        use_noise=False,
        num_visual_predicates=None,
    )
