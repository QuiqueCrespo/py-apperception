import random
import os
import sys

# -------------------------------------- Flags ----------------------------------

# If task type is prediction, then the hidden sensory readings
# are placed at the end of the sequence.
#
# If task type is retrodiction, then the hidden sensory readings
# are placed at the beginning of the sequence.
#
# If task type is imputation, then the hidden sensory readings
# are placed randomly within the sequence.

class TaskType:
    """Base class for task types."""
    def __str__(self):
        if self.__class__.__name__.lower().replace('task', '') == 'prediction':
            return "predict"
        elif self.__class__.__name__.lower().replace('task', '') == 'retrodiction':
            return "retrodict"
        elif self.__class__.__name__.lower().replace('task', '') == 'imputation':
            return "impute"
        

class TaskPrediction(TaskType):
    """Represents a prediction task."""
    pass

class TaskRetrodiction(TaskType):
    """Represents a retrodiction task."""
    pass

class TaskImputation(TaskType):
    """Represents an imputation task."""
    pass

# -------------------------------------- Random selection -----------------------

def hidden_pairs(task_type, max_i, max_j):
    """
    Determines and returns a list of (time, sensor_index) pairs
    that should be hidden, based on the task type.

    Args:
        task_type (TaskType): The type of task (Prediction, Retrodiction, Imputation).
        max_i (int): The maximum time index (inclusive).
        max_j (int): The maximum sensor index (inclusive).

    Returns:
        list: A list of (int, int) tuples representing the hidden pairs.
    """
    if isinstance(task_type, TaskPrediction):
        return [(max_i, j) for j in range(1, max_j + 1)]
    elif isinstance(task_type, TaskRetrodiction):
        return [(1, j) for j in range(1, max_j + 1)]
    elif isinstance(task_type, TaskImputation):
        # In Haskell, 'n' for random_pairs is max_j.
        # This implies that for imputation, it picks 'max_j' number of random pairs.
        return random_pairs(max_i, max_j, max_j)
    else:
        raise ValueError("Unknown TaskType")

def random_pairs(max_i, max_j, n):
    """Return ``n`` unique ``(time, sensor)`` pairs.

    The previous implementation repeatedly constructed the list of all
    possible pairs for every sample which was wasteful.  Here we build the
    Cartesian product once and rely on :func:`random.sample` to efficiently
    pick distinct pairs.

    Parameters
    ----------
    max_i : int
        Maximum time index (inclusive).
    max_j : int
        Maximum sensor index (inclusive).
    n : int
        Number of unique pairs to return.

    Returns
    -------
    list[tuple[int, int]]
        The selected pairs.
    """

    all_pairs = [(i, j) for i in range(1, max_i + 1)
                 for j in range(1, max_j + 1)]
    if n > len(all_pairs):
        raise ValueError("Requested more pairs than available")
    return random.sample(all_pairs, n)

def random_indices(initial_seed, xs):
    """
    Creates a list of random indices, with ranges specified by xs.
    Note: The Haskell version uses a RandomGen parameter 'r' explicitly.
    In Python, we'll use the global random state for simplicity, or
    you could pass a `random.Random` instance if you need a specific seed.
    For consistency with the rest of the translated code, we'll keep
    the `random.Random` instance for now, although `random.seed()` could be used.
    """
    r = random.Random(initial_seed) # Use a specific seed for reproducibility if needed
    return _random_indices_recursive(r, xs)

def _random_indices_recursive(r_gen, xs):
    """
    Recursive helper for random_indices to mimic Haskell's (r, [a]) return type.
    """
    if not xs:
        return r_gen, []
    
    x = xs[0]
    rest_xs = xs[1:]

    # Pick a random number within [1, x] using the provided generator
    y = r_gen.randint(1, x)
    
    r_gen_after_pick, ys = _random_indices_recursive(r_gen, rest_xs)
    
    return r_gen_after_pick, [y] + ys

# This is just for demonstration if you wanted to run RandomPairs directly.
# The main ECA module will import and use these functions.
if __name__ == "__main__":
    print("--- Testing TaskPrediction ---")
    prediction_hidden = hidden_pairs(TaskPrediction(), 5, 3)
    print(f"Hidden pairs for prediction (max_i=5, max_j=3): {prediction_hidden}")
    
    print("\n--- Testing TaskRetrodiction ---")
    retrodiction_hidden = hidden_pairs(TaskRetrodiction(), 5, 3)
    print(f"Hidden pairs for retrodiction (max_i=5, max_j=3): {retrodiction_hidden}")

    print("\n--- Testing TaskImputation ---")
    # Imputation will generate 'max_j' random pairs
    imputation_hidden = hidden_pairs(TaskImputation(), 5, 3)
    print(f"Hidden pairs for imputation (max_i=5, max_j=3, n=max_j): {imputation_hidden}")

    print("\n--- Testing random_indices ---")
    # For random_indices, you might want to use a specific seed for reproducible results
    seed = 42
    initial_gen = random.Random(seed)
    final_gen, rand_idx = random_indices(seed, [10, 5, 8])
    print(f"Random indices with seed {seed} for [10, 5, 8]: {rand_idx}")
    
    # Demonstrate that the generator state is passed correctly (though less explicit in Python)
    # If you wanted to continue generating from 'final_gen', you'd pass its state.
    # For now, it's illustrative that 'random.Random' objects maintain their state.