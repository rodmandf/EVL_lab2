"""Повторное создание координат для варианта 12."""

import csv
import random
from pathlib import Path


rng = random.Random(52723)
points = [(0, 50, 50)]
for number in range(1, 21):
    points.append((number, round(rng.uniform(5, 95), 2),
                   round(rng.uniform(5, 95), 2)))

path = Path(__file__).parent / "data" / "points.csv"
with path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(("id", "x", "y"))
    writer.writerows(points)
print(path)
