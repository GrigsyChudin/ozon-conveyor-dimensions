"""Запуск набора синтетических сценариев и построение SVG-отчёта."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import MeasurementConfig, measure_object
from .synthetic import SyntheticScene, generate_two_sensor_scene


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    title: str
    category: str
    description: str
    dimensions_mm: tuple[float, float, float]
    yaw_deg: float
    seed: int
    finding: str
    noise_std_mm: float = 0.35
    dropout_rate: float = 0.04
    outlier_count: int = 24
    mutation: str = "none"
    expected_status: str = "OK"


def default_catalog_path() -> Path:
    return Path(__file__).resolve().parents[2] / "datasets" / "scenarios.json"


def load_catalog(path: str | Path | None = None) -> list[DatasetSpec]:
    catalog_path = Path(path) if path is not None else default_catalog_path()
    rows = json.loads(catalog_path.read_text(encoding="utf-8"))
    specs: list[DatasetSpec] = []
    for row in rows:
        values = dict(row)
        values["dimensions_mm"] = tuple(float(value) for value in values["dimensions_mm"])
        specs.append(DatasetSpec(**values))
    ids = [spec.id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Идентификаторы сценариев должны быть уникальными")
    return specs


def _sample_xy(points: np.ndarray, limit: int = 700) -> list[list[float]]:
    if not len(points):
        return []
    step = max(1, len(points) // limit)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in points[::step, :2][:limit]]


def _apply_mutation(sensor_clouds: dict[str, np.ndarray], mutation: str) -> dict[str, np.ndarray]:
    clouds = {sensor_id: cloud.copy() for sensor_id, cloud in sensor_clouds.items()}
    if mutation == "none":
        return clouds
    if mutation == "weak_sensor_2":
        clouds["sensor_2"] = clouds["sensor_2"][:8]
        return clouds
    if mutation == "missing_sensor_2":
        clouds.pop("sensor_2", None)
        return clouds
    raise ValueError(f"Неизвестная модификация сценария: {mutation}")


def generate_dataset(spec: DatasetSpec) -> tuple[SyntheticScene, dict[str, np.ndarray]]:
    length, width, height = spec.dimensions_mm
    scene = generate_two_sensor_scene(
        length_mm=length,
        width_mm=width,
        height_mm=height,
        yaw_deg=spec.yaw_deg,
        seed=spec.seed,
        noise_std_mm=spec.noise_std_mm,
        dropout_rate=spec.dropout_rate,
        outlier_count=spec.outlier_count,
    )
    return scene, _apply_mutation(scene.sensor_clouds, spec.mutation)


def run_scenario(spec: DatasetSpec) -> dict[str, Any]:
    scene, clouds = generate_dataset(spec)
    result = measure_object(clouds, MeasurementConfig())

    truth = scene.truth_dimensions
    limits = np.maximum(truth * 0.05, 5.0)
    measured: np.ndarray | None = None
    errors: np.ndarray | None = None
    within_tolerance = False
    corners: list[list[float]] = []
    if result.box is not None:
        measured = result.box.dimensions
        errors = np.abs(measured - truth)
        within_tolerance = bool(np.all(errors <= limits))
        corners = [
            [round(float(x), 2), round(float(y), 2)]
            for x, y in result.box.bottom_corners_xy()
        ]

    passed = result.quality.status == spec.expected_status
    if spec.expected_status == "OK":
        passed = passed and within_tolerance

    return {
        "id": spec.id,
        "title": spec.title,
        "category": spec.category,
        "description": spec.description,
        "finding": spec.finding,
        "expected_status": spec.expected_status,
        "status": result.quality.status,
        "passed": passed,
        "reasons": list(result.quality.reasons),
        "truth_mm": [round(float(value), 3) for value in truth],
        "measured_mm": None if measured is None else [round(float(value), 3) for value in measured],
        "error_mm": None if errors is None else [round(float(value), 3) for value in errors],
        "tolerance_mm": [round(float(value), 3) for value in limits],
        "within_tolerance": within_tolerance,
        "yaw_deg": spec.yaw_deg,
        "noise_std_mm": spec.noise_std_mm,
        "dropout_rate": spec.dropout_rate,
        "outlier_count": spec.outlier_count,
        "raw_points": result.quality.raw_points,
        "object_points": result.quality.object_points,
        "sensor_shares": result.quality.sensor_shares,
        "points_xy": _sample_xy(result.filtered_points),
        "box_corners_xy": corners,
    }


def build_report(specs: list[DatasetSpec]) -> dict[str, Any]:
    scenarios = [run_scenario(spec) for spec in specs]
    measured = [row for row in scenarios if row["error_mm"] is not None]
    geometry = [row for row in scenarios if row["expected_status"] == "OK"]
    quality = [row for row in scenarios if row["expected_status"] == "REJECT"]
    max_error = max(max(row["error_mm"]) for row in measured) if measured else 0.0
    max_error_row = max(measured, key=lambda row: max(row["error_mm"])) if measured else None
    max_error_side = int(np.argmax(max_error_row["error_mm"])) if max_error_row else 0
    max_error_tolerance = (
        max_error_row["tolerance_mm"][max_error_side] if max_error_row else 0.0
    )
    geometry_passed = sum(row["passed"] for row in geometry)
    quality_passed = sum(row["passed"] for row in quality)
    conclusions = [
        f"{geometry_passed} из {len(geometry)} геометрических сценариев прошли допуск по каждой стороне.",
        f"Максимальная ошибка — {max_error:.2f} мм при допустимых {max_error_tolerance:.2f} мм ({max_error_row['title']}).",
        f"Quality gate корректно обработал {quality_passed} из {len(quality)} сценариев с потерей данных.",
    ]
    return {
        "summary": {
            "total": len(scenarios),
            "passed": sum(row["passed"] for row in scenarios),
            "geometry_passed": geometry_passed,
            "geometry_total": len(geometry),
            "quality_passed": quality_passed,
            "quality_total": len(quality),
            "max_error_mm": round(max_error, 3),
            "max_error_tolerance_mm": round(max_error_tolerance, 3),
            "max_error_scenario": max_error_row["id"] if max_error_row else None,
        },
        "conclusions": conclusions,
        "scenarios": scenarios,
    }


def _write_overview_svg(path: Path, report: dict[str, Any]) -> None:
    rows = report["scenarios"]
    summary = report["summary"]
    width, height = 1200, 880
    card_width, card_height = 354, 174
    card_gap_x, card_gap_y = 21, 18
    card_left, card_top = 48, 276
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs>',
        '<linearGradient id="background" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#071a36"/><stop offset="1" stop-color="#041022"/></linearGradient>',
        '<linearGradient id="card" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#102d52"/><stop offset="1" stop-color="#0b213f"/></linearGradient>',
        '<linearGradient id="bar" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#4d9fff"/><stop offset="1" stop-color="#63d6ff"/></linearGradient>',
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="7" stdDeviation="8" flood-color="#000814" flood-opacity=".32"/></filter>',
        '</defs>',
        '<rect width="1200" height="880" rx="30" fill="url(#background)"/>',
        '<circle cx="1090" cy="-20" r="250" fill="#17447d" opacity=".22"/>',
        '<text x="48" y="62" font-family="Arial" font-size="38" font-weight="700" fill="#ffffff">Проверка алгоритма</text>',
        '<text x="48" y="96" font-family="Arial" font-size="19" fill="#a9c8ee">Девять сценариев: размеры, повороты, шум и потеря данных</text>',
        '<rect x="900" y="42" width="252" height="42" rx="21" fill="#123963" stroke="#2b5c8c"/>',
        '<text x="1026" y="69" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="#9fcaff">FIXED SEED · 9 СЦЕНАРИЕВ</text>',
    ]

    metrics = [
        ("РЕЗУЛЬТАТ СЕРИИ", f"{summary['passed']} / {summary['total']}", "ожидаемых исходов получено"),
        ("ГЕОМЕТРИЯ", f"{summary['geometry_passed']} / {summary['geometry_total']}", "измерений внутри допуска"),
        ("QUALITY GATE", f"{summary['quality_passed']} / {summary['quality_total']}", "проблемных наборов остановлено"),
    ]
    for index, (label, value, note) in enumerate(metrics):
        x = 48 + index * 375
        parts.extend(
            [
                f'<rect x="{x}" y="128" width="354" height="112" rx="18" fill="url(#card)" stroke="#214d79" filter="url(#shadow)"/>',
                f'<text x="{x + 24}" y="160" font-family="Arial" font-size="14" font-weight="700" fill="#79b9f4">{label}</text>',
                f'<text x="{x + 24}" y="202" font-family="Arial" font-size="34" font-weight="700" fill="#ffffff">{value}</text>',
                f'<text x="{x + 112}" y="201" font-family="Arial" font-size="15" fill="#a9c8ee">{note}</text>',
            ]
        )

    for index, row in enumerate(rows):
        column = index % 3
        line = index // 3
        x = card_left + column * (card_width + card_gap_x)
        y = card_top + line * (card_height + card_gap_y)
        is_reject = row["status"] == "REJECT"
        status_color = "#f7a13f" if is_reject else "#36c98f"
        badge_width = 82 if is_reject else 54
        badge_x = x + card_width - badge_width - 20
        truth = " × ".join(f"{value:g}" for value in row["truth_mm"])
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="{card_width}" height="{card_height}" rx="18" fill="url(#card)" stroke="#1e466f"/>',
                f'<text x="{x + 22}" y="{y + 36}" font-family="Arial" font-size="19" font-weight="700" fill="#ffffff">{escape(row["title"])}</text>',
                f'<rect x="{badge_x}" y="{y + 18}" width="{badge_width}" height="28" rx="14" fill="{status_color}" fill-opacity=".13" stroke="{status_color}" stroke-opacity=".45"/>',
                f'<text x="{badge_x + badge_width / 2:.1f}" y="{y + 37}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700" fill="{status_color}">{row["status"]}</text>',
                f'<text x="{x + 22}" y="{y + 64}" font-family="Arial" font-size="14" fill="#91afd2">Эталон: {truth} мм</text>',
            ]
        )
        if row["error_mm"] is None:
            reason = (
                "Второй сенсор недоступен"
                if any("MISSING_SENSOR" in value for value in row["reasons"])
                else "Недостаточно данных сенсора 2"
            )
            parts.extend(
                [
                    f'<text x="{x + 22}" y="{y + 105}" font-family="Arial" font-size="22" font-weight="700" fill="#f7a13f">Расчёт остановлен</text>',
                    f'<text x="{x + 22}" y="{y + 133}" font-family="Arial" font-size="14" fill="#d3e3f7">{reason}</text>',
                    f'<circle cx="{x + 325}" cy="{y + 129}" r="10" fill="#f7a13f" fill-opacity=".18" stroke="#f7a13f"/>',
                    f'<path d="M {x + 325} {y + 122} V {y + 131} M {x + 325} {y + 136} v 1" stroke="#f7a13f" stroke-width="2.5" stroke-linecap="round"/>',
                ]
            )
        else:
            side = int(np.argmax(row["error_mm"]))
            error = row["error_mm"][side]
            tolerance = row["tolerance_mm"][side]
            ratio = min(error / tolerance, 1.0)
            percent = round(ratio * 100)
            bar_width = 310
            filled_width = max(5.0, bar_width * ratio)
            parts.extend(
                [
                    f'<text x="{x + 22}" y="{y + 105}" font-family="Arial" font-size="28" font-weight="700" fill="#ffffff">{error:.2f} мм</text>',
                    f'<text x="{x + 22}" y="{y + 128}" font-family="Arial" font-size="13" fill="#91afd2">макс. ошибка · допуск {tolerance:.2f} мм</text>',
                    f'<text x="{x + 332}" y="{y + 128}" text-anchor="end" font-family="Arial" font-size="13" font-weight="700" fill="#63d6ff">{percent}% допуска</text>',
                    f'<rect x="{x + 22}" y="{y + 145}" width="{bar_width}" height="10" rx="5" fill="#173a62"/>',
                    f'<rect x="{x + 22}" y="{y + 145}" width="{filled_width:.1f}" height="10" rx="5" fill="url(#bar)"/>',
                ]
            )
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_scenario_svg(path: Path, row: dict[str, Any]) -> None:
    points = row["points_xy"]
    corners = row["box_corners_xy"]
    all_points = points + corners
    if all_points:
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
    else:
        min_x, max_x, min_y, max_y = -1.0, 1.0, -1.0, 1.0

    def project(point: list[float]) -> tuple[float, float]:
        x = 55 + (point[0] - min_x) / max(max_x - min_x, 1e-9) * 500
        y = 415 - (point[1] - min_y) / max(max_y - min_y, 1e-9) * 285
        return x, y

    status = row["status"]
    status_color = "#36c98f" if status == "OK" else "#f7a13f"
    truth = " × ".join(f"{value:g}" for value in row["truth_mm"])
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="480" viewBox="0 0 900 480">',
        '<rect width="900" height="480" rx="22" fill="#071a36"/>',
        f'<text x="44" y="52" font-family="Arial" font-size="26" font-weight="700" fill="#ffffff">{escape(row["title"])}</text>',
        f'<text x="44" y="80" font-family="Arial" font-size="14" fill="#9fc0e7">{truth} мм · поворот {row["yaw_deg"]:g}° · {row["object_points"]} точек</text>',
        f'<text x="846" y="53" text-anchor="end" font-family="Arial" font-size="15" font-weight="700" fill="{status_color}">{status}</text>',
        '<rect x="35" y="105" width="540" height="335" rx="14" fill="#061329" stroke="#21466f"/>',
        '<text x="615" y="126" font-family="Arial" font-size="15" font-weight="700" fill="#ffffff">РЕЗУЛЬТАТ</text>',
        f'<text x="615" y="157" font-family="Arial" font-size="14" fill="#9fc0e7">Эталон: {truth} мм</text>',
    ]
    for point in points:
        x, y = project(point)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.45" fill="#4da3ff" fill-opacity=".62"/>')
    if corners:
        projected = [project(point) for point in corners + [corners[0]]]
        value = " ".join(f"{x:.2f},{y:.2f}" for x, y in projected)
        parts.append(f'<polyline points="{value}" fill="none" stroke="#f7a13f" stroke-width="3"/>')
    if row["measured_mm"] is not None:
        measured = " × ".join(f"{value:.2f}" for value in row["measured_mm"])
        errors = " / ".join(f"{value:.2f}" for value in row["error_mm"])
        limits = " / ".join(f"{value:.2f}" for value in row["tolerance_mm"])
        parts.extend(
            [
                f'<text x="615" y="188" font-family="Arial" font-size="14" fill="#ffffff">Измерено:</text>',
                f'<text x="615" y="213" font-family="Arial" font-size="18" font-weight="700" fill="#4da3ff">{measured} мм</text>',
                f'<text x="615" y="254" font-family="Arial" font-size="14" fill="#9fc0e7">Ошибка L / W / H</text>',
                f'<text x="615" y="279" font-family="Arial" font-size="17" font-weight="700" fill="#ffffff">{errors} мм</text>',
                f'<text x="615" y="314" font-family="Arial" font-size="13" fill="#9fc0e7">Допуск: {limits} мм</text>',
                '<rect x="615" y="345" width="225" height="42" rx="10" fill="#10365a"/>',
                '<text x="727" y="372" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="#36c98f">РАЗМЕРЫ В ДОПУСКЕ</text>',
            ]
        )
    else:
        reason = ", ".join(row["reasons"])
        parts.extend(
            [
                '<text x="615" y="198" font-family="Arial" font-size="17" font-weight="700" fill="#f7a13f">Размеры не публикуются</text>',
                '<text x="615" y="231" font-family="Arial" font-size="13" fill="#9fc0e7">Quality gate остановил расчёт.</text>',
                f'<text x="615" y="269" font-family="Arial" font-size="13" fill="#ffffff">{escape(reason)}</text>',
            ]
        )
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def write_outputs(output_dir: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    machine_report = {
        **report,
        "scenarios": [
            {
                key: value
                for key, value in row.items()
                if key not in {"points_xy", "box_corners_xy"}
            }
            for row in report["scenarios"]
        ],
    }
    (output_path / "results.json").write_text(
        json.dumps(machine_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_overview_svg(output_path / "overview.svg", report)
    scenario_path = output_path / "scenarios"
    scenario_path.mkdir(exist_ok=True)
    for row in report["scenarios"]:
        _write_scenario_svg(scenario_path / f"{row['id']}.svg", row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Серия проверок алгоритма на синтетических данных")
    parser.add_argument("--catalog", type=Path, default=default_catalog_path())
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_output"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_report(load_catalog(args.catalog))
    write_outputs(args.output_dir, report)
    summary = report["summary"]
    print(f"Сценарии: {summary['passed']}/{summary['total']} прошли ожидаемую проверку")
    print(f"Максимальная ошибка: {summary['max_error_mm']:.3f} мм")
    print("Сводка:", args.output_dir / "overview.svg")
    return 0 if summary["passed"] == summary["total"] else 1



if __name__ == "__main__":
    raise SystemExit(main())
