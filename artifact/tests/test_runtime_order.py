import ast
from types import SimpleNamespace
import unittest
from artifact.mil_artifact.config import repo_root


class RuntimeOrderTests(unittest.TestCase):
    """Execute only the two tiny list selectors, with no search or GPU imports."""
    @classmethod
    def setUpClass(cls):
        path = repo_root() / 'benchmarks/schedule_multi_model.py'
        names = ['_get_the_first_plan_state_when_sorted_by_gpu_num_and_topology',
                 '_get_the_first_plan_state_when_sorted_by_topology']
        tree = ast.parse(path.read_text())
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
            ast.alias(name='annotations')], level=0)] + selected, type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {}
        exec(compile(module, str(path), 'exec'), namespace)
        cls.gpu_order = staticmethod(namespace[names[0]])
        cls.topology_order = staticmethod(namespace[names[1]])

    def states(self):
        def state(mid, workers, dp, parents):
            return SimpleNamespace(exec_plan=SimpleNamespace(
                model=SimpleNamespace(model_id=mid, input_model_ids=parents),
                num_worker=workers, dp_size=dp))
        return [state(0, 1, 1, []), state(1, 2, 2, []), state(2, 8, 1, [0])]

    def test_gpu_priority_applies_only_to_dependency_roots_and_counts_dp(self):
        states = self.states()
        first, remaining = self.gpu_order(states)
        self.assertIs(first, states[1])
        self.assertEqual(remaining, [states[0], states[2]])
        self.assertEqual(len(states), 3)

    def test_topology_only_selector_preserves_existing_root_order(self):
        states = self.states()
        first, remaining = self.topology_order(states)
        self.assertIs(first, states[0])
        self.assertEqual(remaining, states[1:])
