"""Генерация синтетических данных для демонстрации без реальных датчиков."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np


@dataclass(frozen=True)
class SyntheticScene:
    sensor_clouds: dict[str, np.ndarray]
    truth_dimensions: np.ndarray
    truth_yaw_deg: float


def _axis_samples(size: float, step: float) -> np.ndarray:
    count = max(2, int(round(size / step)) + 1)
    return np.linspace(-size / 2.0, size / 2.0, count)


def _box_surfaces(length: float, width: float, height: float, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Создаёт точки и нормали на шести гранях параллелепипеда."""

    xs = _axis_samples(length, step)
    ys = _axis_samples(width, step)
    zs = np.linspace(0.0, height, max(2, int(round(height / step)) + 1))
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []

    def add_face(a: np.ndarray, b: np.ndarray, fixed_axis: int, fixed_value: float, normal: tuple[float, float, float]):
        aa, bb = np.meshgrid(a, b, indexing="ij")
        face = np.zeros((aa.size, 3), dtype=float)
        free_axes = [axis for axis in range(3) if axis != fixed_axis]
        face[:, free_axes[0]] = aa.ravel()
        face[:, free_axes[1]] = bb.ravel()
        face[:, fixed_axis] = fixed_value
        points.append(face)
        normals.append(np.tile(np.asarray(normal, dtype=float), (len(face), 1)))

    add_face(xs, ys, 2, height, (0, 0, 1))
    add_face(xs, ys, 2, 0.0, (0, 0, -1))
    add_face(ys, zs, 0, length / 2.0, (1, 0, 0))
    add_face(ys, zs, 0, -length / 2.0, (-1, 0, 0))
    add_face(xs, zs, 1, width / 2.0, (0, 1, 0))
    add_face(xs, zs, 1, -width / 2.0, (0, -1, 0))
    return np.vstack(points), np.vstack(normals)


def _rotate_z(values: np.ndarray, yaw_deg: float) -> np.ndarray:
    angle = radians(yaw_deg)
    rotation = np.array(
        [[cos(angle), -sin(angle), 0.0], [sin(angle), cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    return values @ rotation.T


def _belt_points(rng: np.random.Generator, step: float = 12.0) -> np.ndarray:
    xs = np.arange(-300.0, 300.0 + step, step)
    ys = np.arange(-280.0, 280.0 + step, step)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = rng.normal(0.0, 0.18, size=xx.size)
    return np.column_stack([xx.ravel(), yy.ravel(), zz])


def generate_two_sensor_scene(
    length_mm: float = 200.0,
    width_mm: float = 100.0,
    height_mm: float = 40.0,
    yaw_deg: float = 27.0,
    step_mm: float = 5.0,
    seed: int = 7,
) -> SyntheticScene:
    """Имитирует два встречных профилометра, ленту, шум и выбросы."""

    if min(length_mm, width_mm, height_mm) <= 0:
        raise ValueError("Размеры должны быть положительными")

    rng = np.random.default_rng(seed)
    # У реального профилометра шаг около 1–2 мм. Для крупных объектов в демо
    # используется более редкая сетка ради скорости, а для минимального товара
    # 10×10×10 мм шаг автоматически уменьшается.
    effective_step = min(step_mm, max(0.75, min(length_mm, width_mm, height_mm) / 5.0))
    local_points, local_normals = _box_surfaces(length_mm, width_mm, height_mm, effective_step)
    points = _rotate_z(local_points, yaw_deg)
    normals = _rotate_z(local_normals, yaw_deg)

    # Направления от объекта к левому и правому верхнему сенсору.
    view_1 = np.array([-0.342, 0.0, 0.940])
    view_2 = np.array([0.342, 0.0, 0.940])
    clouds: dict[str, np.ndarray] = {}
    belt = _belt_points(rng)

    for sensor_id, view, bias in (
        ("sensor_1", view_1, np.array([0.20, -0.10, 0.15])),
        ("sensor_2", view_2, np.array([-0.15, 0.12, -0.10])),
    ):
        visible = (normals @ view) > 0.05
        object_points = points[visible].copy()
        keep = rng.random(len(object_points)) > 0.04
        object_points = object_points[keep]
        object_points += rng.normal(0.0, 0.35, size=object_points.shape) + bias

        belt_part = belt[rng.random(len(belt)) < 0.45]
        outliers = np.column_stack(
            [
                rng.uniform(-260.0, 260.0, 24),
                rng.uniform(-240.0, 240.0, 24),
                rng.uniform(10.0, 220.0, 24),
            ]
        )
        clouds[sensor_id] = np.vstack([object_points, belt_part, outliers])

    truth = np.array([max(length_mm, width_mm), min(length_mm, width_mm), height_mm], dtype=float)
    return SyntheticScene(sensor_clouds=clouds, truth_dimensions=truth, truth_yaw_deg=float(yaw_deg))
