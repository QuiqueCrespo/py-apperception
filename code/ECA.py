import os
import math
import sys

from RandomPairs import hidden_pairs, TaskPrediction, TaskRetrodiction, TaskImputation

input_prefix = "new_eca_input_r"

# -------------------------------------- Types ----------------------------------

class Bin:
    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.value == other.value

    def __ord__(self):
        return self.value

class On(Bin):
    def __init__(self):
        self.value = 1

    def __repr__(self):
        return "On"

    def __str__(self):
        return "⬛"

class Off(Bin):
    def __init__(self):
        self.value = 0

    def __repr__(self):
        return "Off"

    def __str__(self):
        return "⬜"

# Context is a tuple of Bin, Bin, Bin
# type Context = (Bin, Bin, Bin)

class Binary:
    def __init__(self, bs):
        self.bs = list(bs) # Ensure it's a mutable list

    def __eq__(self, other):
        return isinstance(other, Binary) and self.bs == other.bs

    def __add__(self, other):
        if not isinstance(other, Binary):
            return NotImplemented
        # Concatenate the bit lists from both Binary instances
        return Binary(self.bs + other.bs)

    def __len__(self):
        return len(self.bs)

    def __iter__(self):
        return iter(self.bs)

    def __repr__(self):
        return f"Binary({self.bs})"

    def __str__(self):
        return "".join(str(b) for b in self.bs)

# Rule is a dictionary mapping Context to Bin
# type Rule = Map.Map Context Bin

# -------------------------------------- Step -----------------------------------

def step(r, b):
    # Haskell: cs = map (get_context b) [0 .. blength b - 1]
    cs = [get_context(b, i) for i in range(blength(b))]
    # Haskell: B (map (apply_rule r) cs)
    return Binary([apply_rule(r, c) for c in cs])

def blength(b):
    # Haskell: length bs
    return len(b.bs)

def apply_rule(r, c):
    # Haskell: case Map.lookup c r of Just b -> b; Nothing -> error $ "Map lookup failed for context " ++ show c
    val = r.get(c)
    if val is None:
        raise ValueError(f"Map lookup failed for context {c}")
    return val

def get_context(b, i):
    # Haskell's pattern matching and guards translated to if/elif/else
    length_bs = len(b.bs)
    if i == 0:
        x = b.bs[length_bs - 1]
        y = b.bs[0]
        z = b.bs[1]
    elif i == length_bs - 1:
        x = b.bs[i - 1]
        y = b.bs[i]
        z = b.bs[0]
    else:
        x = b.bs[i - 1]
        y = b.bs[i]
        z = b.bs[i + 1]
    return (x, y, z)

def process_rule(i, b, n):
    # Haskell: process (int_to_rule i) b n
    return process(int_to_rule(i), b, n)

def process(r, b, n):
    # Haskell: b : process2 r b (n - 1)
    return [b] + process2(r, b, n - 1)

def process2(r, b, n):
    # Haskell: process2 _ _ 0 = []; process2 r b n = b' : process2 r b' (n-1) where b' = step r b
    if n == 0:
        return []
    b_prime = step(r, b)
    return [b_prime] + process2(r, b_prime, n - 1)

def int_to_rule(n):
    # Haskell: Map.fromList ps where ps = zip all_contexts bs; bs = convert_to_bin n
    ps = zip(all_contexts, convert_to_bin(n))
    return dict(ps)

all_contexts = [
    (On(), On(), On()),
    (On(), On(), Off()),
    (On(), Off(), On()),
    (On(), Off(), Off()),
    (Off(), On(), On()),
    (Off(), On(), Off()),
    (Off(), Off(), On()),
    (Off(), Off(), Off())
]

def convert_to_bin(n):
    # Haskell: map f s2 where s = showIntAtBase 2 intToDigit n ""; s2 = replicate k '0' ++ s; k = 8 - length s; f '0' = Off; f '1' = On
    s = bin(n)[2:]  # Convert to binary string, remove "0b" prefix
    k = 8 - len(s)
    s2 = '0' * k + s

    def f(char):
        if char == '0':
            return Off()
        elif char == '1':
            return On()
        else:
            raise ValueError(f"Invalid character in binary string: {char}")

    return [f(char) for char in s2]

def print_rule(ri, b, n):
    # Haskell: Monad.forM_ bs print
    bs = process_rule(ri, b, n)
    for b_item in bs:
        print(b_item)

k_max_touch = 3

def touch_sensor_readings(tss, bs):
    # Haskell: decay_touch_sensor_readings tsrs where tsrs = map f bs; f (B xs) = map (g xs) tss; g xs ts = case xs !! ts of On -> k_max_touch; Off -> 0
    tsrs = []
    for b_item in bs:
        xs = b_item.bs
        tsr_row = []
        for ts_val in tss:
            if xs[ts_val] == On():
                tsr_row.append(k_max_touch)
            else:
                tsr_row.append(0)
        tsrs.append(tsr_row)
    return decay_touch_sensor_readings(tsrs)

def decay_touch_sensor_readings(tsrs):
    # Haskell: decay_touch_sensor_readings2 tsrs xs where xs = replicate n 0; n = length (head tsrs)
    n = len(tsrs[0]) if tsrs else 0
    xs = [0] * n
    return decay_touch_sensor_readings2(tsrs, xs)

def decay_touch_sensor_readings2(x_list, y_list):
    # Haskell: decay_touch_sensor_readings2 [] _ = []; decay_touch_sensor_readings2 (x:xs) y = (x':xs') where x' = map f (zip x y); xs' = decay_touch_sensor_readings2 xs x'; f (n1, n2) | n2 > n1 = n2 - 1; f (n1, n2) | otherwise = n1
    if not x_list:
        return []

    x_head = x_list[0]
    x_tail = x_list[1:]

    x_prime = []
    for n1, n2 in zip(x_head, y_list):
        if n2 > n1:
            x_prime.append(n2 - 1)
        else:
            x_prime.append(n1)
    
    xs_prime = decay_touch_sensor_readings2(x_tail, x_prime)
    return [x_prime] + xs_prime

def output_rule(dir_path, e):
    # Haskell: let (ri, bi, t, ot, ns) = (rule_index e, start_index e, num_times e, num_observed_times e, num_sensors e)
    ri = e.rule_index
    bi = e.start_index
    t = e.num_times
    ot = e.num_observed_times
    ns = e.num_sensors

    # Haskell: let b = start_config !! bi
    b = start_config[bi]

    # Haskell: let tt = show (task_type e)
    tt = str(e.task_type)

    # Haskell: let f = dir ++ "/" ++ tt ++ "_" ++ show ri ++ "_b" ++ show bi ++ "_t" ++ show t ++ ".lp"
    f = os.path.join(dir_path, f"{tt}_{ri}_b{bi}_t{t}.lp")

    output_rule2(dir_path, f, ri, bi, t, ot, ns, b, e.touch_sensors, e.task_type)

def output_rule2(dir_path, f, ri, bi, t, ot, ns, b, touch_ss, task_type):
    # Haskell: let bs = process_rule ri b t
    bs = process_rule(ri, b, t)

    print(f"Using rule {ri}")
    print(f"Using initial configuration: {b}")
    print("")
    print(f"Generating {f}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(f), exist_ok=True)

    with open(f, "w") as file:
        file.write("%------------------------------------------------------------------------------\n")
        file.write(f"% This file was generated using rule {ri}\n")
        file.write(f"% with configuration {b}\n")
        file.write("%------------------------------------------------------------------------------\n")
        file.write("\n")
        file.write("%------------------------------------------------------------------------------\n")
        file.write("% The sensory given\n")
        file.write("%------------------------------------------------------------------------------\n")
        file.write("\n")
        file.write("%------------------------------------------------------------------------------\n")
        file.write("% Time  State\n")
        file.write("%\n")
        # Monad.forM_ (zip [1..] bs) $ \(i, b) -> appendFile f $ "% " ++ pad_int i ++ "\t" ++ show b ++ "\n"
        for i, b_item in enumerate(bs, 1):
            file.write(f"% {pad_int(i)}\t{b_item}\n")
        file.write("%------------------------------------------------------------------------------\n")
        file.write("\n")
        file.write("% The given sequence\n")
        # ps <- hidden_pairs task_type (length bs) ns
        ps = hidden_pairs(task_type, len(bs), ns)
        # Monad.forM_ (zip [1..] bs) $ \(i, B b) -> do Monad.forM_ (zip [1..ns] b) $ \(j, x) -> do ...
        for i, binary_obj in enumerate(bs, 1):
            for j, x in enumerate(binary_obj.bs, 1):
                p = "c_on" if x == On() else "c_off"
                prd = "hidden" if (i, j) in ps else "senses"
                file.write(f"{prd}(s({p}, obj_cell_{j}), {i}).\n")
        file.write("\n")

        # case touch_ss of [] -> return (); ts -> do ...
        if touch_ss:
            tsrs = touch_sensor_readings(touch_ss, bs)
            file.write("% The touch sensors\n")
            # Monad.forM_ (zip [1..] tsrs) $ \(i, tsr) -> do Monad.forM_ (zip [1..ns] tsr) $ \(j, x) -> do ...
            for i, tsr_row in enumerate(tsrs, 1):
                for j, x in enumerate(tsr_row, 1):
                    # Check if (i, (touch_ss !! (j-1)) + 1 ) is in ps
                    # Python lists are 0-indexed, so touch_ss[j-1]
                    prd = "hidden" if (i, touch_ss[j-1] + 1) in ps else "senses"
                    file.write(f"{prd}(s2(c_touch, obj_touch_sensor_{j}, obj_touch_{x}), {i}).\n")
            file.write("\n")

        file.write("% Elements\n")
        nobjs = min(blength(b), ns)
        objs = [f"obj_cell_{i}" for i in range(1, nobjs + 1)]
        # Monad.forM_ objs $ \x -> appendFile f $ "is_object(" ++ x ++ ").\n"
        for x in objs:
            file.write(f"is_object({x}).\n")

        touch_sensor_objs = [f"obj_touch_sensor_{i}" for i in range(1, len(touch_ss) + 1)]
        # case touch_sensor_objs of [] -> return (); _ -> appendFile f $ "is_object(" ++ concat (List.intersperse ";" touch_sensor_objs) ++ ").\n"
        if touch_sensor_objs:
            file.write(f"is_object({';'.join(touch_sensor_objs)}).\n")

        ts = [str(i) for i in range(1, t + 1)]
        file.write(f"is_time(1..{len(ts)}).\n")
        file.write("is_concept(c_on).\n")
        file.write("is_concept(c_off).\n")
        file.write("\n")
        file.write("% Input exclusions\n")
        file.write("% Every sensor is either on or off\n")
        file.write("% S : sensor → on(S) ⊕ off(S)\n")
        file.write("\n")
        file.write("% At most one\n")
        file.write(":-\n")
        file.write("\tholds(s(c_on, X), T),\n")
        file.write("\tholds(s(c_off, X), T).\n")
        file.write("\n")
        file.write("% At least one\n")
        file.write(":-\n")
        file.write("\tpermanent(isa(t_sensor, X)),\n")
        file.write("\tis_time(T),\n")
        file.write("\tnot holds(s(c_on, X), T),\n")
        file.write("\tnot holds(s(c_off, X), T).\n")
        file.write("\n")
        file.write("% Incompossibility\n")
        file.write("incompossible(s(c_on, X), s(c_off, X)) :-\n")
        file.write("\tpermanent(isa(t_sensor, X)).\n")
        file.write("\n")
        file.write("exclusion_output(\"c_on+c_off\").\n")

        if touch_ss:
            file.write("\n".join(add_exclusions_for_touch_sensors))

add_exclusions_for_touch_sensors = [
    "",
    "% Touch sensor exclusions",
    "",
    "is_concept(c_touch).",
    "",
    "% ∃! clause for c_touch : at most one",
    ":-",
    "\tholds(s2(c_touch, X, Y), T),",
    "\tholds(s2(c_touch, X, Y2), T),",
    "\tY != Y2.",
    "",
    "% ∃! clause for c_touch : at least one",
    ":-",
    "\tpermanent(isa(t_touch_sensor, X)),",
    "\tis_time(T),\n",
    "\tnot aux_c_touch(X, T).",
    "",
    "aux_c_touch(X, T) :-",
    "\tholds(s2(c_touch, X, _), T).",
    "",
    "% Incompossibility for p_r",
    "incompossible(s2(c_touch, X, Y), s2(c_touch, X, Y2)) :-",
    "\tpermanent(isa(t_touch_sensor, X)),",
    "\tpermanent(isa(t_touch, Y)),",
    "\tpermanent(isa(t_touch, Y2)),",
    "\tY != Y2."
]

def pad_int(i):
    # Haskell: Printf.printf "%3d" i
    return f"{i:3d}"

# -------------------------------------- Main ----------------------------------

def main():
    args = sys.argv[1:]
    if args == ["all"]:
        # Monad.forM_ all_examples $ \e -> output_rule "data/eca" e
        for e in all_examples:
            output_rule("data/eca", e)
        write_single_experiment()
    elif args == ["binding"]:
        do_binding_examples()
    else:
        print("Usage: python your_script_name.py [all/binding]")

def do_binding_examples():
    # Monad.forM_ binding_examples $ \e -> output_rule "data/misc" e
    for e in binding_examples:
        output_rule("data/misc", e)
    output_binding_single_experiment()

class Example:
    def __init__(self, rule_index, start_index, num_times, num_observed_times, num_sensors, touch_sensors, task_type):
        self.rule_index = rule_index
        self.start_index = start_index
        self.num_times = num_times
        self.num_observed_times = num_observed_times
        self.num_sensors = num_sensors
        self.touch_sensors = touch_sensors
        self.task_type = task_type

initial_states = [5]
different_times = [14]
all_task_types = [TaskPrediction(), TaskRetrodiction(), TaskImputation()]

all_examples = [
    Example(
        rule_index=i,
        start_index=j,
        num_times=t,
        num_observed_times=t - 1,
        num_sensors=11,
        touch_sensors=[],
        task_type=task
    )
    for i in range(256) # 0 to 255
    for j in initial_states
    for t in different_times
    for task in all_task_types
]

def gen_single_experiment():
    # Haskell: hs ++ xs ++ ts where hs = ["#!/bin/bash", "", "case $(expr $1 + 1) in"]; ts = ["esac"]
    # Haskell: ps = [(i, j, t) | i <- [0..255], j <- initial_states, t <- different_times]
    # Haskell: xs = concat (map f (zip [1..] ps))
    # Haskell: f (n, (i, j, t)) = ["\t" ++ show n ++ " )", "\t\techo \"Solving eca r" ++ show i ++ ", b" ++ show j ++ ", t" ++ show t ++ "...\"", "\t\ttime ./solve eca " ++ input_prefix ++ show i ++ "_b" ++ show j ++ "_t" ++ show t ++ ".lp", "\t\t;;"]
    hs = ["#!/bin/bash", "", "case $(expr $1 + 1) in"]
    ts = ["esac"]
    ps = [(i, j, t) for i in range(256) for j in initial_states for t in different_times]

    xs = []
    for n, (i, j, t) in enumerate(ps, 1):
        xs.append(f"\t{n} )")
        xs.append(f"\t\techo \"Solving eca r{i}, b{j}, t{t}...\"")
        xs.append(f"\t\ttime ./solve eca {input_prefix}{i}_b{j}_t{t}.lp")
        xs.append("\t\t;;")
    return hs + xs + ts

def write_single_experiment():
    # Haskell: let f = "single_eca_experiment.sh"; writeFile f (unlines gen_single_experiment); let c = "chmod 777 " ++ f; Process.callCommand c
    f = "single_eca_experiment.sh"
    with open(f, "w") as file:
        file.write("\n".join(gen_single_experiment()))
    os.chmod(f, 0o777) # Equivalent to chmod 777

def output_binding_single_experiment():
    # Haskell: let f = "single_binding_experiment.sh"; writeFile f (unlines gen_binding_experiment); let c = "chmod 777 " ++ f; Process.callCommand c
    f = "single_binding_experiment.sh"
    with open(f, "w") as file:
        file.write("\n".join(gen_binding_experiment()))
    os.chmod(f, 0o777)

def gen_binding_experiment():
    # Haskell: hs ++ xs ++ ts where hs = ["#!/bin/bash", "", "case $(expr $1 + 1) in"]; ts = ["esac"]
    # Haskell: xs = concat (map f (zip [1..] binding_examples))
    # Haskell: f (n, e) = let (i, j) = (rule_index e, start_index e) in ["\t" ++ show n ++ " )", "\t\techo \"Solving binding r" ++ show i ++ ", b" ++ show j ++ "...\"", "\t\ttime ./solve binding " ++ input_prefix ++ show i ++ "_b" ++ show j ++ "_t14.lp", "\t\t;;"]
    hs = ["#!/bin/bash", "", "case $(expr $1 + 1) in"]
    ts = ["esac"]

    xs = []
    for n, e in enumerate(binding_examples, 1):
        i = e.rule_index
        j = e.start_index
        xs.append(f"\t{n} )")
        xs.append(f"\t\techo \"Solving binding r{i}, b{j}...\"")
        xs.append(f"\t\ttime ./solve binding {input_prefix}{i}_b{j}_t14.lp")
        xs.append("\t\t;;")
    return hs + xs + ts

def write_evaluate_baselines():
    # Haskell: let f = "run_baselines_eca.sh"; writeFile f (unlines evaluate_baselines); let c = "chmod 777 " ++ f; Process.callCommand c
    f = "run_baselines_eca.sh"
    with open(f, "w") as file:
        file.write("\n".join(evaluate_baselines))
    os.chmod(f, 0o777)

evaluate_baselines = [
    "echo \"Evaluating baselines for eca...\"",
    "",
    "rm experiments/baselines/eca_*",
    ""
] + [
    f"clingo --warn=no-atom-undefined pure/baselines.lp data/eca/input_r{i}_b{j}.lp > experiments/baselines/eca_r{i}_b{j}.txt"
    for i in range(256) for j in initial_states
] + [
    "echo \"Correct constant baseline :\"",
    "grep \"baseline_k_eca_correct\" experiments/baselines/eca_* | wc -l",
    "echo \"Total eca examples\"",
    "find experiments/baselines/eca_* | wc -l",
    "echo \"Correct inertia baseline :\"",
    "grep \"baseline_inertia_eca_correct\" experiments/baselines/eca_* | wc -l",
    "echo \"Total eca examples\"",
    "find experiments/baselines/eca_* | wc -l"
]

def output_eca_ilasp_examples():
    # Haskell: let ps = zip [2..11] (map init_b [2..11])
    # Haskell: Monad.forM_ ps $ \(i,b) -> output_rule2 "data/eca" ("eca_ilasp_r245_" ++ show i ++ ".lp") 245 i 6 5 i b [] TaskPrediction
    ps = [(i, init_b(i)) for i in range(2, 12)] # [2..11] in Haskell is inclusive
    for i, b in ps:
        output_rule2("data/eca", f"eca_ilasp_r245_{i}.lp", 245, i, 6, 5, i, b, [], TaskPrediction())


# -------------------------------------- Data --------------------------------

# These are various small configurations
b0 = Binary(Off() for _ in range(4)) + Binary([On()]) + Binary(Off() for _ in range(4))
b1 = Binary(Off() for _ in range(20)) + Binary([On()]) + Binary(Off() for _ in range(20))
b2 = Binary(Off() for _ in range(2)) + Binary([On()]) + Binary(Off() for _ in range(2))
b3 = Binary([On(), On(), Off()])
b4 = Binary(Off() for _ in range(9)) + Binary([On()])

# b5 to b14 are the ten configurations of length 11
# These are the configurations used in the experiments.
b5 = Binary(Off() for _ in range(5)) + Binary([On()]) + Binary(Off() for _ in range(5))
b6 = Binary([Off(), Off(), Off(), On(), Off(), Off(), Off(), On(), Off(), Off(), On()])
b7 = Binary([Off(), On(), Off(), On(), Off(), On(), Off(), On(), Off(), On(), Off()])
b8 = Binary([Off(), On(), On(), On(), Off(), On(), Off(), On(), On(), On(), Off()])
b9 = Binary([Off(), Off(), Off(), On(), Off(), Off(), Off(), On(), On(), Off(), Off()])
b10 = Binary([Off(), On(), On(), On(), Off(), On(), Off(), On(), On(), Off(), Off()])
b11 = Binary([On(), Off(), On(), Off(), On(), Off(), On(), Off(), On(), Off(), On()])
b12 = Binary(On() for _ in range(5)) + Binary([Off()]) + Binary(On() for _ in range(5))
b13 = Binary([Off(), On(), Off(), On(), Off(), Off(), Off(), Off(), Off(), Off(), On()])
b14 = Binary([On(), Off(), Off(), On(), Off(), Off(), On(), On(), Off(), Off(), Off()])

start_config = [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14]

def init_b(n):
    # Haskell: init_b 0 = B []; init_b n | n `rem` 2 == 0 = let B x = init_b (n-1) in B (Off : x); init_b n | n `rem` 2 == 1 = let B x = init_b (n-1) in B (On : x)
    if n == 0:
        return Binary([])
    elif n % 2 == 0:
        x_binary = init_b(n - 1)
        return Binary([Off()] + x_binary.bs)
    else: # n % 2 == 1
        x_binary = init_b(n - 1)
        return Binary([On()] + x_binary.bs)

binding_example1 = Example(
    rule_index = 110,
    start_index = 5,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [2, 10],
    task_type = TaskPrediction()
)

binding_example2 = Example(
    rule_index = 2,
    start_index = 5,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [2, 8],
    task_type = TaskPrediction()
)

binding_example3 = Example(
    rule_index = 11,
    start_index = 5,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [1, 5],
    task_type = TaskPrediction()
)

binding_example4 = Example(
    rule_index = 13,
    start_index = 5,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 7],
    task_type = TaskPrediction()
)

binding_example5 = Example(
    rule_index = 25,
    start_index = 5,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 5],
    task_type = TaskPrediction()
)

binding_example6 = Example(
    rule_index = 26,
    start_index = 5,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 4],
    task_type = TaskPrediction()
)

binding_example7 = Example(
    rule_index = 30,
    start_index = 5,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 5],
    task_type = TaskPrediction()
)

binding_example8 = Example(
    rule_index = 61,
    start_index = 5,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 4],
    task_type = TaskPrediction()
)

binding_example9 = Example(
    rule_index = 67,
    start_index = 5,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 5],
    task_type = TaskPrediction()
)

binding_example10 = Example(
    rule_index = 90,
    start_index = 5,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 4],
    task_type = TaskPrediction()
)

binding_example11 = Example(
    rule_index = 133,
    start_index = 5,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 1],
    task_type = TaskPrediction()
)

binding_example12 = Example(
    rule_index = 135,
    start_index = 5,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 4],
    task_type = TaskPrediction()
)

binding_example13 = Example(
    rule_index = 139,
    start_index = 5,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 4],
    task_type = TaskPrediction()
)

binding_example14 = Example(
    rule_index = 147,
    start_index = 6,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 4],
    task_type = TaskPrediction()
)

binding_example15 = Example(
    rule_index = 148,
    start_index = 13,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 4],
    task_type = TaskPrediction()
)

binding_example16 = Example(
    rule_index = 155,
    start_index = 8,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 4],
    task_type = TaskPrediction()
)

binding_example17 = Example(
    rule_index = 158,
    start_index = 14,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 4],
    task_type = TaskPrediction()
)

binding_example18 = Example(
    rule_index = 167,
    start_index = 11,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 4],
    task_type = TaskPrediction()
)

binding_example19 = Example(
    rule_index = 176,
    start_index = 13,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 4],
    task_type = TaskPrediction()
)

binding_example20 = Example(
    rule_index = 193,
    start_index = 9,
    num_times = 14,
    num_observed_times = 13,
    num_sensors = 11,
    touch_sensors = [0, 1],
    task_type = TaskPrediction()
)

binding_examples = [
    binding_example1, binding_example2, binding_example3, binding_example4,
    binding_example5, binding_example6, binding_example7, binding_example8,
    binding_example9, binding_example10, binding_example11, binding_example12,
    binding_example13, binding_example14, binding_example15, binding_example16,
    binding_example17, binding_example18, binding_example19, binding_example20
]

if __name__ == "__main__":
    main()