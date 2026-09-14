"""Командная строка демонстрационного проекта."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .pipeline import MeasurementConfig, measure_object
from .synthetic import generate_two_sensor_scene
from .visualization import write_demo_svg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Демо измерения габаритов товара на конвейере")
    parser.add_argument("--length", type=float, default=200.0, help="длина эталона, мм")
    parser.add_argument("--width", type=float, default=100.0, help="ширина эталона, мм")
    parser.add_argument("--height", type=float, default=40.0, help="высота эталона, мм")
    parser.add_argument("--yaw", type=float, default=27.0, help="поворот на ленте, градусы")
    parser.add_argument("--seed", type=int, default=7, help="seed генератора шума")
    parser.add_argument("--output-dir", type=Path, default=Path("demo_output"))
    return parser


def _write_filtered_csv(path: Path, points: np.ndarray, labels: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x_mm", "y_mm", "z_mm", "sensor_id"])
        for point, label in zip(points, labels):
            writer.writerow([f"{point[0]:.4f}", f"{point[1]:.4f}", f"{point[2]:.4f}", label])


def run(args: argparse.Namespace) -> int:
    scene = generate_two_sensor_scene(
        length_mm=args.length,
        width_mm=args.width,
        height_mm=args.height,
        yaw_deg=args.yaw,
        seed=args.seed,
    )
    result = measure_object(scene.sensor_clouds, MeasurementConfig())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = result.to_dict()
    payload["truth_dimensions_mm"] = [round(float(v), 3) for v in scene.truth_dimensions]
    payload["truth_yaw_deg"] = scene.truth_yaw_deg

    if result.box is not None:
        measured = result.box.dimensions
        errors = np.abs(measured - scene.truth_dimensions)
        limits = np.maximum(scene.truth_dimensions * 0.05, 5.0)
        payload["absolute_error_mm"] = [round(float(v), 3) for v in errors]
        payload["tolerance_mm"] = [round(float(v), 3) for v in limits]
        payload["within_tolerance"] = bool(np.all(errors <= limits))
        write_demo_svg(
            args.output_dir / "demo.svg",
            raw_clouds=scene.sensor_clouds,
            filtered_points=result.filtered_points,
            box=result.box,
            truth_dimensions=scene.truth_dimensions,
        )

    (args.output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_filtered_csv(args.output_dir / "filtered_points.csv", result.filtered_points, result.filtered_sensor_ids)

    print("Статус:", result.quality.status)
    if result.box is not None:
        print("Эталон L/W/H, мм:   ", " / ".join(f"{x:.2f}" for x in scene.truth_dimensions))
        print("Измерено L/W/H, мм: ", " / ".join(f"{x:.2f}" for x in result.box.dimensions))
        print("Абс. ошибка, мм:     ", " / ".join(f"{x:.2f}" for x in errors))
        print("В допуске:", "ДА" if payload["within_tolerance"] else "НЕТ")
        print("SVG:", args.output_dir / "demo.svg")
    else:
        print("Причины:", ", ".join(result.quality.reasons))
    print("JSON:", args.output_dir / "result.json")
    return 0 if result.quality.status == "OK" and payload.get("within_tolerance", False) else 1


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

