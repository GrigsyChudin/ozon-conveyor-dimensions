"""Экспорт воспроизводимых синтетических наборов в CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .benchmark import DatasetSpec, default_catalog_path, generate_dataset, load_catalog


def _write_cloud(path: Path, points: np.ndarray) -> None:
    np.savetxt(
        path,
        points,
        delimiter=",",
        header="x_mm,y_mm,z_mm",
        comments="",
        fmt="%.4f",
    )


def generate_datasets(specs: list[DatasetSpec], output_dir: str | Path) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": 1, "coordinate_system": "belt_xyz_mm", "datasets": []}

    for spec in specs:
        scene, clouds = generate_dataset(spec)
        dataset_path = output_path / spec.id
        dataset_path.mkdir(exist_ok=True)
        # При повторном запуске удаляем только файлы сенсоров, созданные этим
        # генератором. Иначе после смены сценария на missing_sensor_2 мог бы
        # сохраниться устаревший sensor_2.csv от предыдущего запуска.
        for sensor_id in ("sensor_1", "sensor_2"):
            stale_path = dataset_path / f"{sensor_id}.csv"
            if stale_path.exists():
                stale_path.unlink()
        sensor_files: dict[str, str] = {}
        for sensor_id, points in sorted(clouds.items()):
            filename = f"{sensor_id}.csv"
            _write_cloud(dataset_path / filename, points)
            sensor_files[sensor_id] = filename

        metadata = {
            "id": spec.id,
            "title": spec.title,
            "description": spec.description,
            "truth_dimensions_mm": [float(value) for value in scene.truth_dimensions],
            "truth_yaw_deg": scene.truth_yaw_deg,
            "seed": spec.seed,
            "noise_std_mm": spec.noise_std_mm,
            "dropout_rate": spec.dropout_rate,
            "outlier_count": spec.outlier_count,
            "mutation": spec.mutation,
            "expected_status": spec.expected_status,
            "sensor_files": sensor_files,
            "point_counts": {sensor_id: len(points) for sensor_id, points in clouds.items()},
        }
        (dataset_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["datasets"].append(
            {
                "id": spec.id,
                "path": spec.id,
                "expected_status": spec.expected_status,
                "sensor_files": sensor_files,
            }
        )

    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Генератор CSV-наборов облаков точек")
    parser.add_argument("--catalog", type=Path, default=default_catalog_path())
    parser.add_argument("--output-dir", type=Path, default=Path("generated_datasets"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = generate_datasets(load_catalog(args.catalog), args.output_dir)
    print(f"Создано наборов: {len(manifest['datasets'])}")
    print("Manifest:", args.output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
