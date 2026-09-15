"""Запуск набора синтетических сценариев и построение HTML-отчёта."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import MeasurementConfig, measure_object
from .synthetic import generate_two_sensor_scene


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


def run_scenario(spec: DatasetSpec) -> dict[str, Any]:
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
    clouds = _apply_mutation(scene.sensor_clouds, spec.mutation)
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
    width = 1200
    row_height = 54
    height = 150 + row_height * len(rows)
    paired_limits = []
    for row in rows:
        if row["error_mm"] is not None:
            side = int(np.argmax(row["error_mm"]))
            paired_limits.append(row["tolerance_mm"][side])
    max_value = max(paired_limits + [1.0])
    chart_left, chart_width = 500, 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="24" fill="#071a36"/>',
        '<text x="48" y="58" font-family="Arial" font-size="30" font-weight="700" fill="#ffffff">Проверка на девяти наборах данных</text>',
        '<text x="48" y="92" font-family="Arial" font-size="16" fill="#a9c8ee">Максимальная ошибка сценария относительно допустимого порога</text>',
        '<text x="1080" y="92" text-anchor="end" font-family="Arial" font-size="14" fill="#a9c8ee">ошибка / допуск, мм</text>',
    ]
    for index, row in enumerate(rows):
        y = 130 + index * row_height
        is_reject = row["status"] == "REJECT"
        status_color = "#f7a13f" if is_reject else "#36c98f"
        status_text = "REJECT" if is_reject else "OK"
        parts.append(
            f'<text x="48" y="{y + 25}" font-family="Arial" font-size="16" fill="#ffffff">{row["title"]}</text>'
        )
        parts.append(
            f'<text x="360" y="{y + 25}" font-family="Arial" font-size="13" font-weight="700" fill="{status_color}">{status_text}</text>'
        )
        if row["error_mm"] is None:
            parts.append(f'<line x1="{chart_left}" y1="{y + 20}" x2="{chart_left + chart_width}" y2="{y + 20}" stroke="#25466f" stroke-width="10" stroke-linecap="round"/>')
            parts.append(f'<text x="{chart_left + 12}" y="{y + 25}" font-family="Arial" font-size="13" fill="#f7c482">измерение остановлено quality gate</text>')
        else:
            side = int(np.argmax(row["error_mm"]))
            error = row["error_mm"][side]
            tolerance = row["tolerance_mm"][side]
            tolerance_x = chart_left + chart_width * tolerance / max_value
            error_width = max(4.0, chart_width * error / max_value)
            parts.append(f'<line x1="{chart_left}" y1="{y + 20}" x2="{chart_left + chart_width}" y2="{y + 20}" stroke="#17385f" stroke-width="10" stroke-linecap="round"/>')
            parts.append(f'<line x1="{chart_left}" y1="{y + 20}" x2="{chart_left + error_width}" y2="{y + 20}" stroke="#4da3ff" stroke-width="10" stroke-linecap="round"/>')
            parts.append(f'<line x1="{tolerance_x}" y1="{y + 9}" x2="{tolerance_x}" y2="{y + 31}" stroke="#f7a13f" stroke-width="3"/>')
            parts.append(f'<text x="1080" y="{y + 25}" text-anchor="end" font-family="Arial" font-size="14" fill="#dcecff">{error:.2f} / {tolerance:.2f}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_dashboard(path: Path, report: dict[str, Any]) -> None:
    data = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = DASHBOARD_HTML.replace("__REPORT_DATA__", data)
    path.write_text(html, encoding="utf-8")


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
    _write_dashboard(output_path / "dashboard.html", report)


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
    print("Dashboard:", args.output_dir / "dashboard.html")
    return 0 if summary["passed"] == summary["total"] else 1


DASHBOARD_HTML = r'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Проверка алгоритма на наборах данных</title>
<style>
:root{color-scheme:light dark;--bg:#06152e;--panel:#0d2343;--panel2:#122d54;--text:#eef6ff;--muted:#9cb9dc;--line:#264b77;--blue:#55a9ff;--orange:#ff9c42;--green:#42d39e;--red:#ff6f7f;--shadow:0 18px 45px rgba(0,0,0,.28)}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#153e75 0,transparent 36%),var(--bg);color:var(--text);font:15px/1.5 Inter,system-ui,-apple-system,Segoe UI,sans-serif;min-height:100vh}.shell{max-width:1240px;margin:auto;padding:38px 24px 56px;overflow:hidden}header{display:flex;gap:20px;justify-content:space-between;align-items:flex-end;margin-bottom:28px}h1{font-size:clamp(28px,4vw,48px);line-height:1.05;margin:0 0 10px;max-width:780px}header p{margin:0;color:var(--muted);font-size:17px}.stamp{white-space:nowrap;color:var(--orange);font-size:13px;font-weight:700;letter-spacing:.04em}.summary{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-bottom:26px}.stat{min-width:0;background:linear-gradient(145deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:18px;padding:18px 20px;box-shadow:var(--shadow)}.stat b{display:block;font-size:clamp(22px,2.5vw,30px);line-height:1.1;margin-top:5px;white-space:nowrap}.stat span{color:var(--muted)}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 16px}.filter,.scenario{font:inherit;color:var(--text);border:1px solid var(--line);background:transparent;cursor:pointer}.filter{border-radius:999px;padding:8px 14px}.filter[aria-pressed="true"]{background:var(--text);color:var(--bg);border-color:var(--text)}.scenarios{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-bottom:24px}.scenario{text-align:left;border-radius:14px;padding:13px 14px;min-height:78px;overflow-wrap:anywhere;transition:transform .18s,border-color .18s,background .18s}.scenario:hover{transform:translateY(-2px);border-color:var(--blue)}.scenario[aria-pressed="true"]{background:var(--panel2);border-color:var(--blue)}.scenario strong,.scenario small{display:block}.scenario small{color:var(--muted);margin-top:2px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;background:var(--green)}.dot.reject{background:var(--orange)}.detail{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(300px,.75fr);gap:18px}.panel{min-width:0;background:rgba(13,35,67,.88);border:1px solid var(--line);border-radius:20px;padding:20px;box-shadow:var(--shadow)}.panel-head{display:flex;justify-content:space-between;gap:18px;align-items:flex-start;margin-bottom:12px}.panel h2,.panel h3{margin:0}.panel h2{font-size:24px}.meta{color:var(--muted);margin:4px 0 0}.badge{border-radius:999px;padding:6px 10px;font-size:12px;font-weight:800;letter-spacing:.05em}.badge.ok{background:rgba(66,211,158,.16);color:var(--green)}.badge.reject{background:rgba(255,156,66,.16);color:var(--orange)}#cloud{display:block;width:100%;height:auto;aspect-ratio:16/9;border-radius:14px;background:linear-gradient(180deg,#07162c,#0a1c35)}#cloud circle{fill:var(--blue);opacity:.65}#cloud polyline{fill:none;stroke:var(--orange);stroke-width:3;vector-effect:non-scaling-stroke}.finding{margin:14px 0 0;padding-left:14px;border-left:3px solid var(--orange);color:#dcecff}.metric{margin:17px 0}.metric-row{display:flex;justify-content:space-between;gap:12px;margin-bottom:7px}.metric-row span:last-child{font-variant-numeric:tabular-nums;color:var(--muted)}.track{height:9px;border-radius:8px;background:#19375d;overflow:hidden;position:relative}.truth,.measured{position:absolute;left:0;height:100%;border-radius:8px}.truth{background:var(--line);top:0}.measured{background:var(--blue);top:0;height:4px;margin-top:2.5px}.legend{display:flex;gap:16px;color:var(--muted);font-size:12px;margin-top:13px}.key{display:inline-block;width:16px;height:5px;border-radius:4px;margin-right:6px;vertical-align:middle;background:var(--blue)}.key.truth{position:static;display:inline-block;background:var(--line);height:9px}.quality{margin-top:18px;padding-top:15px;border-top:1px solid var(--line);color:var(--muted)}.conclusions{margin-top:18px}.conclusions ol{margin:10px 0 0;padding-left:22px}.conclusions li{margin:8px 0}.hidden{display:none!important}@media(max-width:820px){header{display:block}.stamp{display:block;margin-top:12px}.summary{grid-template-columns:1fr}.scenarios{grid-template-columns:1fr 1fr}.detail{grid-template-columns:1fr}}@media(max-width:520px){.shell{padding:24px 14px}.scenarios{grid-template-columns:1fr}.panel{padding:15px}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<div class="shell">
  <header><div><h1>Как алгоритм ведёт себя на разных данных</h1><p>Размеры, повороты, шум и потеря одного из ракурсов.</p></div><span class="stamp">9 НАБОРОВ · FIXED SEED</span></header>
  <section class="summary" aria-label="Итоговые показатели">
    <div class="stat"><span>Пройдено сценариев</span><b id="total-pass"></b></div>
    <div class="stat"><span>Геометрия в допуске</span><b id="geometry-pass"></b></div>
    <div class="stat"><span>Максимальная ошибка</span><b id="max-error"></b></div>
  </section>
  <div class="toolbar" role="group" aria-label="Фильтр сценариев">
    <button class="filter" data-filter="all" aria-pressed="true">Все</button>
    <button class="filter" data-filter="geometry" aria-pressed="false">Размеры и повороты</button>
    <button class="filter" data-filter="robustness" aria-pressed="false">Шум</button>
    <button class="filter" data-filter="quality" aria-pressed="false">Quality gate</button>
  </div>
  <nav class="scenarios" id="scenario-list" aria-label="Наборы данных"></nav>
  <main class="detail" aria-live="polite">
    <section class="panel">
      <div class="panel-head"><div><h2 id="scenario-title"></h2><p class="meta" id="scenario-meta"></p></div><span id="status" class="badge"></span></div>
      <svg id="cloud" viewBox="0 0 640 360" role="img" aria-label="Проекция очищенного облака точек и найденный прямоугольник"></svg>
      <p class="finding" id="finding"></p>
    </section>
    <aside class="panel">
      <h3>Размеры и ошибка</h3>
      <div id="metrics"></div>
      <div class="legend"><span><i class="key truth"></i>эталон</span><span><i class="key"></i>измерение</span></div>
      <p class="quality" id="quality"></p>
    </aside>
  </main>
  <section class="panel conclusions"><h3>Выводы по серии</h3><ol id="conclusions"></ol></section>
</div>
<script>
const report=__REPORT_DATA__;
const labels=['Длина','Ширина','Высота'];
let selected=report.scenarios[0].id;
const list=document.getElementById('scenario-list');
document.getElementById('total-pass').textContent=`${report.summary.passed} / ${report.summary.total}`;
document.getElementById('geometry-pass').textContent=`${report.summary.geometry_passed} / ${report.summary.geometry_total}`;
document.getElementById('max-error').textContent=`${report.summary.max_error_mm.toFixed(2)} / ${report.summary.max_error_tolerance_mm.toFixed(2)} мм`;
document.getElementById('conclusions').innerHTML=report.conclusions.map(x=>`<li>${x}</li>`).join('');
function makeButtons(filter='all'){
  list.innerHTML='';
  report.scenarios.filter(s=>filter==='all'||s.category===filter).forEach(s=>{
    const b=document.createElement('button');b.className='scenario';b.dataset.id=s.id;b.setAttribute('aria-pressed',s.id===selected);
    b.innerHTML=`<strong><i class="dot ${s.status==='REJECT'?'reject':''}"></i>${s.title}</strong><small>${s.description}</small>`;
    b.addEventListener('click',()=>{selected=s.id;render(s);[...list.children].forEach(x=>x.setAttribute('aria-pressed',x.dataset.id===selected));});list.appendChild(b);
  });
  if(!list.querySelector(`[data-id="${selected}"]`)){const first=list.firstElementChild;if(first){selected=first.dataset.id;render(report.scenarios.find(s=>s.id===selected));first.setAttribute('aria-pressed','true')}}
}
function renderCloud(s){
  const svg=document.getElementById('cloud'),all=[...s.points_xy,...s.box_corners_xy];svg.innerHTML='';
  if(!all.length){svg.innerHTML='<text x="320" y="185" text-anchor="middle" fill="#9cb9dc">Нет точек для отображения</text>';return}
  const xs=all.map(p=>p[0]),ys=all.map(p=>p[1]),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys),pad=28;
  const sx=x=>pad+(x-minX)/Math.max(maxX-minX,1)*(640-pad*2),sy=y=>360-pad-(y-minY)/Math.max(maxY-minY,1)*(360-pad*2);
  s.points_xy.forEach((p,i)=>{const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',sx(p[0]));c.setAttribute('cy',sy(p[1]));c.setAttribute('r','1.7');c.style.animation=`point-in .28s ${Math.min(i,80)*2}ms both`;svg.appendChild(c)});
  if(s.box_corners_xy.length){const poly=document.createElementNS('http://www.w3.org/2000/svg','polyline');const pts=[...s.box_corners_xy,s.box_corners_xy[0]].map(p=>`${sx(p[0])},${sy(p[1])}`).join(' ');poly.setAttribute('points',pts);svg.appendChild(poly)}
}
function render(s){
  document.getElementById('scenario-title').textContent=s.title;document.getElementById('scenario-meta').textContent=`${s.truth_mm.join(' × ')} мм · поворот ${s.yaw_deg}° · ${s.object_points} точек`;
  const badge=document.getElementById('status');badge.textContent=s.status;badge.className=`badge ${s.status.toLowerCase()}`;document.getElementById('finding').textContent=s.finding;renderCloud(s);
  const metrics=document.getElementById('metrics');
  if(!s.measured_mm){metrics.innerHTML='<p class="meta">Размеры не публикуются: проверка качества остановила расчёт.</p>'}
  else{const scale=Math.max(...s.truth_mm,...s.measured_mm);metrics.innerHTML=labels.map((label,i)=>`<div class="metric"><div class="metric-row"><strong>${label}</strong><span>${s.measured_mm[i].toFixed(2)} / ${s.truth_mm[i].toFixed(2)} мм · Δ ${s.error_mm[i].toFixed(2)}</span></div><div class="track"><i class="truth" style="width:${s.truth_mm[i]/scale*100}%"></i><i class="measured" style="width:${s.measured_mm[i]/scale*100}%"></i></div></div>`).join('')}
  const reason=s.reasons.length?`Причина: ${s.reasons.join(', ')}`:`Доля сенсоров: ${Object.entries(s.sensor_shares).map(([k,v])=>`${k} ${(v*100).toFixed(1)}%`).join(' · ')}`;document.getElementById('quality').textContent=reason;
}
document.querySelectorAll('.filter').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.filter').forEach(x=>x.setAttribute('aria-pressed','false'));b.setAttribute('aria-pressed','true');makeButtons(b.dataset.filter)}));
const style=document.createElement('style');style.textContent='@keyframes point-in{from{opacity:0;transform:translateY(5px)}to{opacity:.65;transform:none}}';document.head.appendChild(style);
makeButtons();render(report.scenarios[0]);
</script>
</body>
</html>'''


if __name__ == "__main__":
    raise SystemExit(main())
