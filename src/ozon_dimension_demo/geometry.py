"""Базовые геометрические алгоритмы для облака точек.

В прототипе используется OBB с вертикальной осью Z. Это естественное
ограничение для товара, который лежит на горизонтальной ленте. Поворот в
плоскости XY выбирается точно по выпуклой оболочке методом вращающихся
калиперов. Для полной произвольной 3D-ориентации в production-версии можно
заменить эту функцию на Open3D MINIMAL_JYLANKI.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin

import numpy as np


@dataclass(frozen=True)
class OrientedBox:
    """Ориентированный параллелепипед в системе координат конвейера."""

    center: np.ndarray
    length: float
    width: float
    height: float
    yaw_rad: float

    @property
    def dimensions(self) -> np.ndarray:
        return np.array([self.length, self.width, self.height], dtype=float)

    @property
    def volume(self) -> float:
        return float(self.length * self.width * self.height)

    def bottom_corners_xy(self) -> np.ndarray:
        """Четыре угла прямоугольника в плоскости ленты."""

        u = np.array([cos(self.yaw_rad), sin(self.yaw_rad)])
        v = np.array([-sin(self.yaw_rad), cos(self.yaw_rad)])
        half_l = self.length / 2.0
        half_w = self.width / 2.0
        center_xy = self.center[:2]
        return np.array(
            [
                center_xy - half_l * u - half_w * v,
                center_xy + half_l * u - half_w * v,
                center_xy + half_l * u + half_w * v,
                center_xy - half_l * u + half_w * v,
            ]
        )


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_hull_2d(points_xy: np.ndarray) -> np.ndarray:
    """Строит 2D-выпуклую оболочку алгоритмом монотонной цепи.

    Возвращает вершины против часовой стрелки без повторения первой вершины.
    """

    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy должен иметь форму (N, 2)")

    unique = np.unique(points, axis=0)
    if len(unique) < 3:
        raise ValueError("Для выпуклой оболочки нужны минимум три разные точки")

    order = np.lexsort((unique[:, 1], unique[:, 0]))
    sorted_points = unique[order]

    lower: list[np.ndarray] = []
    for point in sorted_points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[np.ndarray] = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1])


def _normalize_yaw(yaw: float) -> float:
    """Нормализует угол в диапазон [-pi/2, pi/2)."""

    while yaw >= pi / 2:
        yaw -= pi
    while yaw < -pi / 2:
        yaw += pi
    return yaw


def minimum_area_rectangle(points_xy: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Находит минимальный прямоугольник вокруг набора 2D-точек.

    Для каждого ребра выпуклой оболочки проверяется система координат, в
    которой это ребро является осью X. Минимальный прямоугольник обязательно
    имеет сторону, параллельную одному из рёбер оболочки.

    Returns:
        center_xy, length, width, yaw_rad
    """

    hull = convex_hull_2d(points_xy)
    best: tuple[float, np.ndarray, float, float, float] | None = None

    for index in range(len(hull)):
        edge = hull[(index + 1) % len(hull)] - hull[index]
        norm = float(np.linalg.norm(edge))
        if norm < 1e-12:
            continue

        u = edge / norm
        v = np.array([-u[1], u[0]])
        x = hull @ u
        y = hull @ v
        min_x, max_x = float(x.min()), float(x.max())
        min_y, max_y = float(y.min()), float(y.max())
        extent_x, extent_y = max_x - min_x, max_y - min_y
        area = extent_x * extent_y
        center = u * ((min_x + max_x) / 2.0) + v * ((min_y + max_y) / 2.0)
        yaw = atan2(float(u[1]), float(u[0]))

        if best is None or area < best[0]:
            best = (area, center, extent_x, extent_y, yaw)

    if best is None:
        raise ValueError("Не удалось построить прямоугольник")

    _, center, extent_x, extent_y, yaw = best
    if extent_y > extent_x:
        extent_x, extent_y = extent_y, extent_x
        yaw += pi / 2

    return center, float(extent_x), float(extent_y), _normalize_yaw(yaw)


def constrained_obb(points_xyz: np.ndarray, belt_z: float = 0.0) -> OrientedBox:
    """Строит OBB, у которого ось высоты перпендикулярна ленте."""

    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz должен иметь форму (N, 3)")
    if len(points) < 4:
        raise ValueError("Для OBB нужны минимум четыре точки")
    if not np.isfinite(points).all():
        raise ValueError("Облако содержит NaN или бесконечность")

    center_xy, length, width, yaw = minimum_area_rectangle(points[:, :2])
    top_z = float(points[:, 2].max())
    height = top_z - float(belt_z)
    if height <= 0:
        raise ValueError("Все точки лежат не выше плоскости ленты")

    center = np.array([center_xy[0], center_xy[1], belt_z + height / 2.0])
    return OrientedBox(center=center, length=length, width=width, height=height, yaw_rad=yaw)


def radius_outlier_mask(points_xyz: np.ndarray, radius_mm: float = 4.0, min_neighbors: int = 3) -> np.ndarray:
    """Возвращает маску точек, имеющих соседей в заданном радиусе.

    Для ускорения пространство разбивается на кубические ячейки, поэтому поиск
    соседей ограничивается ближайшими ячейками пространственной сетки.
    """

    points = np.asarray(points_xyz, dtype=float)
    if radius_mm <= 0:
        raise ValueError("radius_mm должен быть положительным")
    if min_neighbors < 1:
        raise ValueError("min_neighbors должен быть не меньше 1")

    cells = np.floor(points / radius_mm).astype(np.int64)
    grid: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(cells):
        grid.setdefault(tuple(int(value) for value in cell), []).append(index)

    keep = np.zeros(len(points), dtype=bool)
    radius_sq = radius_mm * radius_mm
    offsets = (-1, 0, 1)
    for index, (point, cell) in enumerate(zip(points, cells)):
        neighbor_count = 0
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    key = (int(cell[0] + dx), int(cell[1] + dy), int(cell[2] + dz))
                    for other_index in grid.get(key, ()):
                        if other_index == index:
                            continue
                        delta = points[other_index] - point
                        if float(delta @ delta) <= radius_sq:
                            neighbor_count += 1
                            if neighbor_count >= min_neighbors:
                                keep[index] = True
                                break
                    if keep[index]:
                        break
                if keep[index]:
                    break
            if keep[index]:
                break
    return keep
