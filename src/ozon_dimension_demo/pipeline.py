"""Последовательность обработки данных двух профилометров."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import degrees

import numpy as np

from .geometry import OrientedBox, constrained_obb, radius_outlier_mask


@dataclass(frozen=True)
class MeasurementConfig:
    belt_z_mm: float = 0.0
    belt_tolerance_mm: float = 2.5
    outlier_radius_mm: float = 5.0
    min_neighbors: int = 2
    min_object_points: int = 100
    min_sensor_share: float = 0.10
    max_height_mm: float = 320.0
    uncertainty_mm: float = 3.1


@dataclass(frozen=True)
class QualityInfo:
    status: str
    reasons: tuple[str, ...]
    raw_points: int
    object_points: int
    sensor_shares: dict[str, float]


@dataclass(frozen=True)
class MeasurementResult:
    box: OrientedBox | None
    quality: QualityInfo
    filtered_points: np.ndarray
    filtered_sensor_ids: np.ndarray
    uncertainty_mm: float

    def to_dict(self) -> dict:
        payload: dict = {
            "status": self.quality.status,
            "quality": asdict(self.quality),
            "uncertainty_mm": self.uncertainty_mm,
        }
        if self.box is not None:
            payload["dimensions_mm"] = {
                "length": round(self.box.length, 3),
                "width": round(self.box.width, 3),
                "height": round(self.box.height, 3),
            }
            payload["center_mm"] = [round(float(v), 3) for v in self.box.center]
            payload["yaw_deg"] = round(degrees(self.box.yaw_rad), 3)
            payload["volume_mm3"] = round(self.box.volume, 3)
        return payload


def _combine(sensor_clouds: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not sensor_clouds:
        raise ValueError("Не переданы данные сенсоров")
    clouds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for sensor_id, cloud in sensor_clouds.items():
        points = np.asarray(cloud, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Облако {sensor_id} должно иметь форму (N, 3)")
        clouds.append(points)
        labels.append(np.full(len(points), sensor_id, dtype=object))
    return np.vstack(clouds), np.concatenate(labels)


def measure_object(sensor_clouds: dict[str, np.ndarray], config: MeasurementConfig | None = None) -> MeasurementResult:
    """Удаляет ленту/выбросы, проверяет качество и вычисляет габариты."""

    cfg = config or MeasurementConfig()
    points, sensor_ids = _combine(sensor_clouds)
    raw_count = len(points)

    finite = np.isfinite(points).all(axis=1)
    points, sensor_ids = points[finite], sensor_ids[finite]

    object_mask = points[:, 2] > cfg.belt_z_mm + cfg.belt_tolerance_mm
    object_points = points[object_mask]
    object_sensor_ids = sensor_ids[object_mask]

    if len(object_points) > 0:
        neighbor_mask = radius_outlier_mask(
            object_points,
            radius_mm=cfg.outlier_radius_mm,
            min_neighbors=cfg.min_neighbors,
        )
        object_points = object_points[neighbor_mask]
        object_sensor_ids = object_sensor_ids[neighbor_mask]

    reasons: list[str] = []
    if len(object_points) < cfg.min_object_points:
        reasons.append("TOO_FEW_POINTS")

    shares: dict[str, float] = {}
    if len(object_points):
        for sensor_id in sensor_clouds:
            share = float(np.mean(object_sensor_ids == sensor_id))
            shares[sensor_id] = round(share, 4)
            if share < cfg.min_sensor_share:
                reasons.append(f"LOW_SHARE_{sensor_id}")
        if float(object_points[:, 2].max()) > cfg.max_height_mm:
            reasons.append("HEIGHT_OUT_OF_RANGE")

    box: OrientedBox | None = None
    if not reasons:
        try:
            box = constrained_obb(object_points, belt_z=cfg.belt_z_mm)
        except ValueError:
            reasons.append("OBB_FAILED")

    quality = QualityInfo(
        status="OK" if not reasons else "REJECT",
        reasons=tuple(reasons),
        raw_points=raw_count,
        object_points=len(object_points),
        sensor_shares=shares,
    )
    return MeasurementResult(
        box=box,
        quality=quality,
        filtered_points=object_points,
        filtered_sensor_ids=object_sensor_ids,
        uncertainty_mm=cfg.uncertainty_mm,
    )

