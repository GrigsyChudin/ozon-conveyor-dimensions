"""Простая SVG-визуализация без matplotlib."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np

from .geometry import OrientedBox


def _project(points: np.ndarray, x_index: int, y_index: int, box: tuple[float, float, float, float]) -> np.ndarray:
    left, top, width, height = box
    x = points[:, x_index]
    y = points[:, y_index]
    min_x, max_x = float(x.min()), float(x.max())
    min_y, max_y = float(y.min()), float(y.max())
    pad_x = max(1.0, (max_x - min_x) * 0.06)
    pad_y = max(1.0, (max_y - min_y) * 0.06)
    min_x, max_x = min_x - pad_x, max_x + pad_x
    min_y, max_y = min_y - pad_y, max_y + pad_y
    sx = left + (x - min_x) / max(max_x - min_x, 1e-9) * width
    sy = top + height - (y - min_y) / max(max_y - min_y, 1e-9) * height
    return np.column_stack([sx, sy])


def write_demo_svg(
    path: str | Path,
    raw_clouds: dict[str, np.ndarray],
    filtered_points: np.ndarray,
    box: OrientedBox,
    truth_dimensions: np.ndarray,
) -> None:
    path = Path(path)
    raw = np.vstack(list(raw_clouds.values()))
    raw_sample = raw[:: max(1, len(raw) // 2500)]
    clean_sample = filtered_points[:: max(1, len(filtered_points) // 2200)]

    canvas_w, canvas_h = 1200, 720
    panels = [(55, 110, 510, 430), (635, 110, 510, 430)]
    raw_xy = _project(raw_sample, 0, 1, panels[0])

    # Правая панель масштабируется по точкам и углам найденного бокса вместе.
    corners = box.bottom_corners_xy()
    right_all = np.vstack([clean_sample[:, :2], corners])
    right_projected = _project(np.column_stack([right_all, np.zeros(len(right_all))]), 0, 1, panels[1])
    clean_xy = right_projected[: len(clean_sample)]
    corner_xy = right_projected[len(clean_sample) :]

    measured = box.dimensions
    errors = np.abs(measured - truth_dimensions)
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">',
        '<rect width="100%" height="100%" fill="#f5f7fb"/>',
        '<text x="55" y="46" font-family="Arial" font-size="26" font-weight="700" fill="#172033">Демо измерения габаритов</text>',
        '<text x="55" y="75" font-family="Arial" font-size="15" fill="#667085">Синтетические данные двух лазерных профилометров</text>',
    ]
    for x, y, w, h in panels:
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="white" stroke="#d6deeb"/>')
    svg.extend([
        '<text x="75" y="142" font-family="Arial" font-size="18" font-weight="700" fill="#124896">1. Сырые точки: товар, лента и выбросы</text>',
        '<text x="655" y="142" font-family="Arial" font-size="18" font-weight="700" fill="#124896">2. После очистки + найденный OBB</text>',
    ])

    for x, y in raw_xy:
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.15" fill="#8390a7" fill-opacity="0.55"/>')
    for x, y in clean_xy:
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.35" fill="#005bff" fill-opacity="0.58"/>')
    corner_path = ' '.join(f'{x:.2f},{y:.2f}' for x, y in np.vstack([corner_xy, corner_xy[0]]))
    svg.append(f'<polyline points="{corner_path}" fill="none" stroke="#e79b26" stroke-width="4"/>')

    truth_text = ' × '.join(f'{value:.1f}' for value in truth_dimensions)
    measured_text = ' × '.join(f'{value:.1f}' for value in measured)
    error_text = ' / '.join(f'{value:.1f}' for value in errors)
    svg.extend([
        '<rect x="55" y="575" width="1090" height="102" rx="12" fill="#eaf1ff" stroke="#b8ccf6"/>',
        f'<text x="80" y="610" font-family="Arial" font-size="17" fill="#172033"><tspan font-weight="700">Эталон:</tspan> {escape(truth_text)} мм</text>',
        f'<text x="80" y="640" font-family="Arial" font-size="17" fill="#172033"><tspan font-weight="700">Измерено:</tspan> {escape(measured_text)} мм</text>',
        f'<text x="630" y="610" font-family="Arial" font-size="17" fill="#172033"><tspan font-weight="700">Ошибка L/W/H:</tspan> {escape(error_text)} мм</text>',
        f'<text x="630" y="640" font-family="Arial" font-size="17" fill="#172033"><tspan font-weight="700">Поворот:</tspan> {box.yaw_rad * 180.0 / np.pi:.1f}°</text>',
        '</svg>',
    ])
    path.write_text('\n'.join(svg), encoding='utf-8')

