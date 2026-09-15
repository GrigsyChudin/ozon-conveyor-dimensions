"""Прототип измерения габаритов товара по облаку точек."""

from .geometry import OrientedBox, constrained_obb
from .pipeline import MeasurementConfig, MeasurementResult, measure_object

__all__ = [
    "OrientedBox",
    "constrained_obb",
    "MeasurementConfig",
    "MeasurementResult",
    "measure_object",
]
