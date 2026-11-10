"""Лабораторная 2: маршрут доставки с ограничением времени."""

import csv
import json
import math
import random
import statistics
from pathlib import Path


HERE = Path(__file__).parent


def load_points(path):
    with path.open(encoding="utf-8", newline="") as file:
        return {int(row["id"]): (float(row["x"]), float(row["y"]))
                for row in csv.DictReader(file)}


def distance_table(points):
    return {(a, b): math.dist(points[a], points[b])
            for a in points for b in points}


def route_metrics(route, points, distances, service_minutes):
    """Возвращает длину маршрута и время с учётом обслуживания точек."""
    path = [0] + list(route) + [0]
    length = 0.0
    minutes = len(route) * service_minutes
    for a, b in zip(path, path[1:]):
        edge = distances[a, b]
        length += edge
        # Переезд между левой и правой частью карты занимает на 25% больше.
        if (points[a][0] - 50) * (points[b][0] - 50) < 0:
            minutes += 1.25 * edge
        else:
            minutes += edge
    return length, minutes


def valid_permutation(route, customers):
    return sorted(route) == list(range(1, customers + 1))


class Evaluator:
    def __init__(self, points, distances, config):
        self.points = points
        self.distances = distances
        self.config = config
        self.calls = 0
        self.feasible_calls = 0
        self.best_route = None
        self.best_length = math.inf
        self.best_minutes = math.inf
        self.history = []

    def evaluate(self, route):
        length, minutes = route_metrics(
            route, self.points, self.distances, self.config["service_minutes"])
        self.calls += 1
        if minutes <= self.config["time_limit_minutes"]:
            self.feasible_calls += 1
            if length < self.best_length:
                self.best_route = route.copy()
                self.best_length = length
                self.best_minutes = minutes
        if self.calls % 100 == 0:
            self.history.append(self.best_length)
        return length, minutes


def nearest_neighbor(distances, customers, rng=None, choices=1):
    remaining = set(range(1, customers + 1))
    route = []
    current = 0
    while remaining:
        nearby = sorted(remaining, key=lambda city: (distances[current, city], city))
        current = nearby[0] if rng is None else rng.choice(nearby[:choices])
        route.append(current)
        remaining.remove(current)
    return route


def tournament(population, config, rng):
    sample = rng.sample(population, config["tournament_size"])
    return min(sample, key=lambda item: fitness(item[1], config))[0]


def fitness(metrics, config):
    length, minutes = metrics
    late = max(0.0, minutes - config["time_limit_minutes"])
    return length + config["penalty_weight"] * late


def ordered_crossover(first, second, rng):
    """OX: копируем отрезок первого родителя, остальное — из второго."""
    size = len(first)
    left, right = sorted(rng.sample(range(size), 2))
    child = [None] * size
    child[left:right + 1] = first[left:right + 1]
    remaining = [city for city in second[right + 1:] + second[:right + 1]
                 if city not in child]
    positions = list(range(right + 1, size)) + list(range(left))
    for position, city in zip(positions, remaining):
        child[position] = city
    return child


def mutate(route, operator, rng):
    left, right = sorted(rng.sample(range(len(route)), 2))
    if operator == "swap":
        route[left], route[right] = route[right], route[left]
    else:
        route[left:right + 1] = reversed(route[left:right + 1])


def repair(route, metrics, evaluator, rng):
    """Пробуем сократить время маршрута инверсией отрезков."""
    for _ in range(evaluator.config["repair_attempts"]):
        if metrics[1] <= evaluator.config["time_limit_minutes"]:
            break
        if evaluator.calls >= evaluator.config["evaluation_budget"]:
            break
        candidate = route.copy()
        mutate(candidate, "inversion", rng)
        candidate_metrics = evaluator.evaluate(candidate)
        if candidate_metrics[1] < metrics[1]:
            route, metrics = candidate, candidate_metrics
    return route, metrics


def run_ga(points, distances, config, seed, constraint_method, mutation_operator):
    rng = random.Random(seed)
    evaluator = Evaluator(points, distances, config)
    customers = config["customers"]
    start = nearest_neighbor(distances, customers)
    population = [(start, evaluator.evaluate(start))]
    for _ in range(config["population_size"] - 1):
        route = rng.sample(range(1, customers + 1), customers)
        population.append((route, evaluator.evaluate(route)))

    generations = 0
    while evaluator.calls < config["evaluation_budget"]:
        elite = min(population, key=lambda item: fitness(item[1], config))
        new_population = [elite]
        while (len(new_population) < config["population_size"] and
               evaluator.calls < config["evaluation_budget"]):
            first = tournament(population, config, rng)
            second = tournament(population, config, rng)
            if rng.random() < config["crossover_probability"]:
                child = ordered_crossover(first, second, rng)
            else:
                child = first.copy()
            if rng.random() < config["mutation_probability"]:
                mutate(child, mutation_operator, rng)
            metrics = evaluator.evaluate(child)
            if constraint_method == "repair" and metrics[1] > config["time_limit_minutes"]:
                child, metrics = repair(child, metrics, evaluator, rng)
            new_population.append((child, metrics))
        if len(new_population) < config["population_size"]:
            break
        population = new_population
        generations += 1
    return evaluator, generations


def run_baseline(points, distances, config, seed):
    rng = random.Random(seed)
    evaluator = Evaluator(points, distances, config)
    customers = config["customers"]
    first = nearest_neighbor(distances, customers)
    evaluator.evaluate(first)
    while evaluator.calls < config["evaluation_budget"]:
        route = nearest_neighbor(distances, customers, rng, choices=2)
        evaluator.evaluate(route)
    return evaluator


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output, histories, best_route, points):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (name, runs) in zip(axes.flat, histories.items()):
        x = range(100, 100 * len(runs[0]) + 1, 100)
        low = [min(run[i] for run in runs) for i in range(len(runs[0]))]
        mean = [statistics.mean(run[i] for run in runs)
                for i in range(len(runs[0]))]
        high = [max(run[i] for run in runs) for i in range(len(runs[0]))]
        ax.plot(x, mean, label="mean")
        ax.fill_between(x, low, high, alpha=0.2, label="min–max")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend()
    for ax in axes[1]:
        ax.set_xlabel("Route evaluations")
    for ax in axes[:, 0]:
        ax.set_ylabel("Best feasible distance")
    fig.tight_layout()
    fig.savefig(output / "convergence.png", dpi=150)
    plt.close(fig)

    path = [0] + best_route + [0]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([points[city][0] for city in path],
            [points[city][1] for city in path], "-o", markersize=5)
    for city, (x, y) in points.items():
        ax.annotate(str(city), (x, y), xytext=(4, 4),
                    textcoords="offset points")
    ax.scatter(*points[0], color="red", s=90, zorder=3, label="Depot")
    ax.set(xlabel="x, km", ylabel="y, km", title="Best delivery route")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "best_route.png", dpi=150)
    plt.close(fig)


def main():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    group_index = {507: 0, 517: 1, 527: 2}[config["group"]]
    variant = 1 + (17 * config["student_number"] + 7 * group_index +
                   config["year"]) % 20
    if variant != 12:
        raise ValueError(f"Программа написана для варианта 12, получен {variant}")

    points = load_points(HERE / "data" / "points.csv")
    if len(points) != config["customers"] + 1:
        raise ValueError("Неверное число точек в points.csv")
    distances = distance_table(points)
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    methods = [
        ("Penalty + inversion", "penalty", "inversion"),
        ("Repair + inversion", "repair", "inversion"),
        ("Penalty + swap", "penalty", "swap"),
    ]
    histories = {}
    rows = []
    overall_best = None

    for name, constraint_method, operator in methods:
        histories[name] = []
        for seed in range(config["first_seed"], config["first_seed"] + config["runs"]):
            evaluator, generations = run_ga(
                points, distances, config, seed, constraint_method, operator)
            histories[name].append(evaluator.history)
            rows.append({"method": name, "seed": seed,
                         "best_distance": evaluator.best_length,
                         "best_minutes": evaluator.best_minutes,
                         "best_route": json.dumps(evaluator.best_route),
                         "evaluations": evaluator.calls,
                         "feasible_evaluations": evaluator.feasible_calls,
                         "generations": generations})
            if overall_best is None or evaluator.best_length < overall_best[0]:
                overall_best = (evaluator.best_length, evaluator.best_minutes,
                                evaluator.best_route.copy(), name, seed)

    baseline_name = "Randomized nearest neighbor"
    histories[baseline_name] = []
    for seed in range(config["first_seed"], config["first_seed"] + config["runs"]):
        evaluator = run_baseline(points, distances, config, seed)
        histories[baseline_name].append(evaluator.history)
        rows.append({"method": baseline_name, "seed": seed,
                     "best_distance": evaluator.best_length,
                     "best_minutes": evaluator.best_minutes,
                     "best_route": json.dumps(evaluator.best_route),
                     "evaluations": evaluator.calls,
                     "feasible_evaluations": evaluator.feasible_calls,
                     "generations": 0})

    summary = {}
    history_rows = []
    for name, runs in histories.items():
        method_rows = [row for row in rows if row["method"] == name]
        lengths = [row["best_distance"] for row in method_rows]
        summary[name] = {
            "best": min(lengths), "mean": statistics.mean(lengths),
            "median": statistics.median(lengths),
            "std": statistics.stdev(lengths), "worst": max(lengths),
            "mean_generations": statistics.mean(
                row["generations"] for row in method_rows),
            "mean_feasible_fraction": statistics.mean(
                row["feasible_evaluations"] / row["evaluations"]
                for row in method_rows),
        }
        for index in range(len(runs[0])):
            values = [run[index] for run in runs]
            history_rows.append({"method": name, "evaluations": (index + 1) * 100,
                                 "minimum": min(values),
                                 "mean": statistics.mean(values),
                                 "maximum": max(values)})

    write_csv(output / "runs.csv", ["method", "seed", "best_distance",
              "best_minutes", "best_route", "evaluations",
              "feasible_evaluations", "generations"], rows)
    write_csv(output / "history.csv", ["method", "evaluations", "minimum",
              "mean", "maximum"], history_rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    length, minutes, route, method, seed = overall_best
    (output / "best_route.json").write_text(json.dumps({
        "method": method, "seed": seed, "distance_km": length,
        "time_minutes": minutes, "route": [0] + route + [0]
    }, indent=2), encoding="utf-8")
    route_rows = [{"order": i, "point": city,
                   "x": points[city][0], "y": points[city][1]}
                  for i, city in enumerate([0] + route + [0])]
    write_csv(output / "best_route.csv", ["order", "point", "x", "y"], route_rows)
    plot_results(output, histories, route, points)

    print(f"V={variant}, customers={config['customers']}, limit={config['time_limit_minutes']} min")
    for name, values in summary.items():
        print(f"{name}: best={values['best']:.2f}, mean={values['mean']:.2f}, "
              f"median={values['median']:.2f}, std={values['std']:.2f}, "
              f"worst={values['worst']:.2f}")
    print(f"Overall best: {length:.2f} km, {minutes:.2f} min, {method}, seed={seed}")


if __name__ == "__main__":
    main()
