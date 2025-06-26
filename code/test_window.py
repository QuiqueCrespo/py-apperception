import sys, types, unittest
from pathlib import Path

# Provide dummy matplotlib to import solve without dependency
sys.modules.setdefault('matplotlib', types.ModuleType('matplotlib'))
sys.modules.setdefault('matplotlib.pyplot', types.ModuleType('matplotlib.pyplot'))

from solve import (
    template_sokoban,
    get_sokoban_data,
    _generate_template_files,
    _collect_asp_files,
    _setup_control,
    _run_incremental_solve,
    WINDOW_SIZE,
)

class WindowLimitTest(unittest.TestCase):
    def test_old_steps_removed(self):
        max_x, max_y, n_blocks = get_sokoban_data("e2d_0_0")
        template = template_sokoban(max_x, max_y, n_blocks)
        temp_dir = "sokoban"
        base_name = "predict_e2d_0_0"
        _generate_template_files(template, temp_dir, base_name)
        files = _collect_asp_files(temp_dir, base_name, template)
        files.extend(["asp/judgement.lp", "asp/constraints.lp", "asp/step.lp"])

        ctl = _setup_control(files)
        ctl.ground([("base", [])])

        models, last, ctl_final = _run_incremental_solve(
            ctl,
            template,
            "data/sokoban",
            "predict_e2d_0_0.lp",
            Path("temp/test_window.txt"),
            [],
            None,
        )

        times = sorted(a.symbol.arguments[0].number for a in ctl_final.symbolic_atoms.by_signature("is_time", 1))
        self.assertLessEqual(len(times), WINDOW_SIZE)
        self.assertGreaterEqual(min(times), max(times) - WINDOW_SIZE + 1)

if __name__ == "__main__":
    unittest.main()
