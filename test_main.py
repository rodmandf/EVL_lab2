import json
import random
import unittest
from pathlib import Path

from main import (HERE, distance_table, load_points, mutate,
                  nearest_neighbor, ordered_crossover, route_metrics,
                  run_ga, valid_permutation)


class RouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points = load_points(HERE / "data" / "points.csv")
        cls.distances = distance_table(cls.points)
        cls.config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))

    def test_valid_and_invalid_routes(self):
        self.assertEqual(len(self.points), 21)
        route = nearest_neighbor(self.distances, 20)
        length, minutes = route_metrics(route, self.points, self.distances, 2)
        self.assertTrue(valid_permutation(route, 20))
        self.assertLess(minutes, 535)
        self.assertGreater(length, 0)
        self.assertFalse(valid_permutation(list(range(1, 20)) + [19], 20))
        _, late_minutes = route_metrics(
            list(range(1, 21)), self.points, self.distances, 2)
        self.assertGreater(late_minutes, 535)

    def test_operators_keep_every_customer_once(self):
        rng = random.Random(123)
        first = list(range(1, 21))
        second = first[::-1]
        for _ in range(100):
            child = ordered_crossover(first, second, rng)
            self.assertTrue(valid_permutation(child, 20))
            for operator in ("swap", "inversion"):
                changed = child.copy()
                mutate(changed, operator, rng)
                self.assertTrue(valid_permutation(changed, 20))

    def test_fixed_budget_and_feasible_answer(self):
        config = self.config.copy()
        config["evaluation_budget"] = 300
        evaluator, _ = run_ga(
            self.points, self.distances, config, 2026, "repair", "inversion")
        repeated, _ = run_ga(
            self.points, self.distances, config, 2026, "repair", "inversion")
        self.assertEqual(evaluator.calls, 300)
        self.assertEqual(len(evaluator.history), 3)
        self.assertEqual(evaluator.best_route, repeated.best_route)
        self.assertEqual(evaluator.history, repeated.history)
        self.assertTrue(valid_permutation(evaluator.best_route, 20))
        length, minutes = route_metrics(
            evaluator.best_route, self.points, self.distances, 2)
        self.assertAlmostEqual(length, evaluator.best_length)
        self.assertAlmostEqual(minutes, evaluator.best_minutes)
        self.assertLessEqual(minutes, 535)


if __name__ == "__main__":
    unittest.main()
