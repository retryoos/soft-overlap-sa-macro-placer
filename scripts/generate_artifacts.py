#!/usr/bin/env python3
"""
Generate data-backed artifacts for the submission README.

Default mode is quick and has no plotting dependency: it writes SVG charts and
JSON from published baselines plus the recorded submission aggregate score. With
--challenge-repo, the script can also load a real IBM benchmark. With
--run-placer it runs the placer and emits placement PNGs and an animated GIF; that mode
requires matplotlib and Pillow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Rectangle

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    animation = None
    Rectangle = None
    HAS_MATPLOTLIB = False


IBM_BENCHMARKS = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17", "ibm18",
]

SA_BASELINES = {
    "ibm01": 1.3166, "ibm02": 1.9072, "ibm03": 1.7401, "ibm04": 1.5037,
    "ibm06": 2.5057, "ibm07": 2.0229, "ibm08": 1.9239, "ibm09": 1.3875,
    "ibm10": 2.1108, "ibm11": 1.7111, "ibm12": 2.8261, "ibm13": 1.9141,
    "ibm14": 2.2750, "ibm15": 2.3000, "ibm16": 2.2337, "ibm17": 3.6726,
    "ibm18": 2.7755,
}

REPLACE_BASELINES = {
    "ibm01": 0.9976, "ibm02": 1.8370, "ibm03": 1.3222, "ibm04": 1.3024,
    "ibm06": 1.6187, "ibm07": 1.4633, "ibm08": 1.4285, "ibm09": 1.1194,
    "ibm10": 1.5009, "ibm11": 1.1774, "ibm12": 1.7261, "ibm13": 1.3355,
    "ibm14": 1.5436, "ibm15": 1.5159, "ibm16": 1.4780, "ibm17": 1.6446,
    "ibm18": 1.7722,
}

SUBMISSION_AVG_PROXY = 1.4734
SUBMISSION_TOTAL_RUNTIME_S = 56.0 * 60.0


@dataclass
class PlacementMetrics:
    proxy: float | None = None
    wirelength: float | None = None
    density: float | None = None
    congestion: float | None = None
    overlaps: int | None = None
    runtime_s: float | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def save_static_score_artifacts(out_dir: Path) -> None:
    names = ["Submission", "RePlAce", "SA"]
    values = [
        SUBMISSION_AVG_PROXY,
        sum(REPLACE_BASELINES[b] for b in IBM_BENCHMARKS) / len(IBM_BENCHMARKS),
        sum(SA_BASELINES[b] for b in IBM_BENCHMARKS) / len(IBM_BENCHMARKS),
    ]
    summary = {
        "submission_avg_proxy": SUBMISSION_AVG_PROXY,
        "submission_total_runtime_s": SUBMISSION_TOTAL_RUNTIME_S,
        "replace_avg_proxy": values[1],
        "sa_avg_proxy": values[2],
        "submission_improvement_vs_sa_pct": (values[2] - SUBMISSION_AVG_PROXY) / values[2] * 100.0,
        "submission_delta_vs_replace_pct": (SUBMISSION_AVG_PROXY - values[1]) / values[1] * 100.0,
    }
    (out_dir / "score_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    save_static_score_svgs(out_dir, names, values)
    if HAS_MATPLOTLIB:
        save_static_score_pngs(out_dir, names, values)


def save_static_score_svgs(out_dir: Path, names: list[str], values: list[float]) -> None:
    max_value = max(values)
    colors = ["#2563eb", "#64748b", "#ef4444"]
    rows = []
    for idx, (name, value, color) in enumerate(zip(names, values, colors)):
        y = 105 + idx * 70
        width = 560 * value / max_value
        rows.append(
            f'<text x="42" y="{y + 22}" class="label">{name}</text>'
            f'<rect x="170" y="{y}" width="{width:.1f}" height="34" fill="{color}"/>'
            f'<text x="{180 + width:.1f}" y="{y + 23}" class="value">{value:.4f}</text>'
        )
    avg_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="860" height="350" viewBox="0 0 860 350">
  <style>.bg{{fill:#f8fafc}}.title{{font:700 24px Arial,sans-serif;fill:#0f172a}}.note{{font:14px Arial,sans-serif;fill:#475569}}.label{{font:16px Arial,sans-serif;fill:#334155}}.value{{font:700 16px Arial,sans-serif;fill:#0f172a}}.axis{{stroke:#334155;stroke-width:2}}</style>
  <rect class="bg" width="860" height="350"/>
  <text x="40" y="45" class="title">Submission vs Published IBM ICCAD04 Baselines</text>
  <text x="40" y="72" class="note">Average proxy cost, lower is better.</text>
  <line x1="170" y1="300" x2="760" y2="300" class="axis"/>
  {"".join(rows)}
  <text x="40" y="330" class="note">Submission score is a local development aggregate; baselines are from the challenge README.</text>
</svg>
"""
    (out_dir / "avg_proxy_comparison.svg").write_text(avg_svg)

    points_sa = []
    points_rep = []
    for idx, bench in enumerate(IBM_BENCHMARKS):
        x = 70 + idx * 42
        points_sa.append(f"{x},{290 - SA_BASELINES[bench] / 3.8 * 220:.1f}")
        points_rep.append(f"{x},{290 - REPLACE_BASELINES[bench] / 3.8 * 220:.1f}")
    y_submission = 290 - SUBMISSION_AVG_PROXY / 3.8 * 220
    labels = "".join(
        f'<text x="{70 + idx * 42}" y="318" class="tick" transform="rotate(45 {70 + idx * 42} 318)">{bench}</text>'
        for idx, bench in enumerate(IBM_BENCHMARKS)
    )
    curve_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="860" height="390" viewBox="0 0 860 390">
  <style>.bg{{fill:#f8fafc}}.title{{font:700 22px Arial,sans-serif;fill:#0f172a}}.tick{{font:11px Arial,sans-serif;fill:#475569}}.legend{{font:14px Arial,sans-serif;fill:#334155}}.axis{{stroke:#334155;stroke-width:2}}.grid{{stroke:#e2e8f0;stroke-width:1}}</style>
  <rect class="bg" width="860" height="390"/>
  <text x="40" y="42" class="title">Published Baselines With Submission Aggregate Reference</text>
  <line x1="55" y1="290" x2="790" y2="290" class="axis"/>
  <line x1="55" y1="65" x2="55" y2="290" class="axis"/>
  <line x1="55" y1="160" x2="790" y2="160" class="grid"/>
  <polyline points="{' '.join(points_sa)}" fill="none" stroke="#ef4444" stroke-width="3"/>
  <polyline points="{' '.join(points_rep)}" fill="none" stroke="#64748b" stroke-width="3"/>
  <line x1="55" y1="{y_submission:.1f}" x2="790" y2="{y_submission:.1f}" stroke="#2563eb" stroke-width="3"/>
  {labels}
  <text x="610" y="74" class="legend">SA baseline</text>
  <text x="610" y="96" class="legend">RePlAce baseline</text>
  <text x="610" y="118" class="legend">Submission aggregate</text>
</svg>
"""
    (out_dir / "baseline_curve_with_submission_avg.svg").write_text(curve_svg)


def save_static_score_pngs(out_dir: Path, names: list[str], values: list[float]) -> None:
    colors = ["#2563eb", "#64748b", "#ef4444"]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ax.barh(names, values, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Average proxy cost, lower is better")
    ax.set_title("Submission vs Published IBM ICCAD04 Baselines")
    ax.grid(axis="x", color="#e2e8f0", linewidth=1)
    ax.set_axisbelow(True)
    for idx, val in enumerate(values):
        ax.text(val + 0.02, idx, f"{val:.4f}", va="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "avg_proxy_comparison.png", bbox_inches="tight")
    plt.close(fig)

    bench = list(range(len(IBM_BENCHMARKS)))
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=160)
    ax.plot(bench, [SA_BASELINES[b] for b in IBM_BENCHMARKS], marker="o", label="SA baseline")
    ax.plot(bench, [REPLACE_BASELINES[b] for b in IBM_BENCHMARKS], marker="o", label="RePlAce baseline")
    ax.axhline(SUBMISSION_AVG_PROXY, color="#2563eb", linewidth=2.4, label="Submission aggregate")
    ax.set_xticks(bench)
    ax.set_xticklabels(IBM_BENCHMARKS, rotation=45, ha="right")
    ax.set_ylabel("Proxy cost")
    ax.set_title("Published Per-Benchmark Baselines With Submission Aggregate Reference")
    ax.grid(color="#e2e8f0", linewidth=1)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_curve_with_submission_avg.png", bbox_inches="tight")
    plt.close(fig)


def add_challenge_to_path(challenge_repo: Path) -> None:
    root = repo_root()
    sys.path.insert(0, str(root))
    sys.path.insert(1, str(challenge_repo))


def load_benchmark(challenge_repo: Path, benchmark_name: str):
    add_challenge_to_path(challenge_repo)
    from macro_place.loader import load_benchmark_from_dir

    bench_dir = challenge_repo / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / benchmark_name
    return load_benchmark_from_dir(str(bench_dir))


def compute_metrics(placement, benchmark, plc) -> PlacementMetrics:
    from macro_place.objective import compute_proxy_cost

    costs = compute_proxy_cost(placement, benchmark, plc)
    return PlacementMetrics(
        proxy=float(costs["proxy_cost"]),
        wirelength=float(costs["wirelength_cost"]),
        density=float(costs["density_cost"]),
        congestion=float(costs["congestion_cost"]),
        overlaps=int(costs["overlap_count"]),
    )


def require_matplotlib() -> None:
    if not HAS_MATPLOTLIB or np is None:
        raise RuntimeError("placement PNG/GIF mode requires numpy, matplotlib, and Pillow; install requirements.txt first")


def render_placement(path: Path, positions: np.ndarray, sizes: np.ndarray, fixed: np.ndarray,
                     canvas_width: float, canvas_height: float, title: str,
                     metrics: PlacementMetrics | None) -> None:
    require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=170)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, canvas_height)
    ax.set_title(title)
    ax.set_xlabel("x (micron)")
    ax.set_ylabel("y (micron)")
    ax.grid(color="#e2e8f0", linewidth=0.7)
    for idx, ((x, y), (w, h)) in enumerate(zip(positions, sizes)):
        color = "#ef4444" if fixed[idx] else "#2563eb"
        ax.add_patch(Rectangle((x - w / 2.0, y - h / 2.0), w, h,
                               facecolor=color, edgecolor="#0f172a",
                               linewidth=0.35, alpha=0.72))
    if metrics is not None:
        lines = []
        if metrics.proxy is not None:
            lines.append(f"proxy {metrics.proxy:.4f}")
        if metrics.overlaps is not None:
            lines.append(f"overlaps {metrics.overlaps}")
        if metrics.runtime_s is not None:
            lines.append(f"runtime {metrics.runtime_s:.1f}s")
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top",
                fontsize=10, bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cbd5e1"))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_component_chart(path: Path, initial: PlacementMetrics, final: PlacementMetrics) -> None:
    require_matplotlib()
    labels = ["wirelength", "density", "congestion"]
    init_values = [initial.wirelength, initial.density, initial.congestion]
    final_values = [final.wirelength, final.density, final.congestion]
    if any(v is None for v in init_values + final_values):
        return
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    ax.bar(x - width / 2, init_values, width, label="initial.plc", color="#94a3b8")
    ax.bar(x + width / 2, final_values, width, label="final output", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cost contribution")
    ax.set_title("Proxy Objective Components")
    ax.grid(axis="y", color="#e2e8f0")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_interpolation_gif(path: Path, start_pos: np.ndarray, end_pos: np.ndarray,
                           sizes: np.ndarray, fixed: np.ndarray, canvas_width: float,
                           canvas_height: float, title: str, frames: int = 36) -> None:
    require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 7), dpi=130)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, canvas_height)
    ax.set_title(title)
    ax.grid(color="#e2e8f0", linewidth=0.7)
    patches = []
    for idx, ((x, y), (w, h)) in enumerate(zip(start_pos, sizes)):
        color = "#ef4444" if fixed[idx] else "#2563eb"
        rect = Rectangle((x - w / 2.0, y - h / 2.0), w, h,
                         facecolor=color, edgecolor="#0f172a", linewidth=0.3, alpha=0.72)
        ax.add_patch(rect)
        patches.append(rect)

    def ease(t: float) -> float:
        return 0.5 - 0.5 * math.cos(math.pi * t)

    def update(frame: int):
        t = ease(frame / max(1, frames - 1))
        pos = start_pos * (1.0 - t) + end_pos * t
        for rect, (x, y), (w, h) in zip(patches, pos, sizes):
            rect.set_xy((x - w / 2.0, y - h / 2.0))
        return patches

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=90, blit=True)
    try:
        ani.save(path, writer=animation.PillowWriter(fps=10))
    finally:
        plt.close(fig)


def run_real_benchmark_artifacts(out_dir: Path, challenge_repo: Path, benchmark_name: str,
                                 run_placer: bool) -> None:
    if not HAS_MATPLOTLIB:
        raise RuntimeError("--challenge-repo placement rendering requires matplotlib; install requirements.txt first")
    original_cwd = Path.cwd()
    os.chdir(challenge_repo)
    try:
        benchmark, plc = load_benchmark(challenge_repo, benchmark_name)
        initial = benchmark.macro_positions.clone()
        initial_metrics = compute_metrics(initial, benchmark, plc)
        hard_n = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:hard_n].detach().cpu().numpy()
        fixed = benchmark.macro_fixed[:hard_n].detach().cpu().numpy()
        initial_np = initial[:hard_n].detach().cpu().numpy()
        render_placement(out_dir / f"{benchmark_name}_initial.png", initial_np, sizes, fixed,
                         benchmark.canvas_width, benchmark.canvas_height,
                         f"{benchmark_name} initial.plc", initial_metrics)
        if not run_placer:
            return
        from placer import SoftOverlapSAPlacer

        placer = SoftOverlapSAPlacer()
        start = time.time()
        final = placer.place(benchmark)
        runtime = time.time() - start
        final_metrics = compute_metrics(final, benchmark, plc)
        final_metrics.runtime_s = runtime
        final_np = final[:hard_n].detach().cpu().numpy()
        render_placement(out_dir / f"{benchmark_name}_final.png", final_np, sizes, fixed,
                         benchmark.canvas_width, benchmark.canvas_height,
                         f"{benchmark_name} final output", final_metrics)
        render_component_chart(out_dir / f"{benchmark_name}_objective_components.png", initial_metrics, final_metrics)
        save_interpolation_gif(out_dir / f"{benchmark_name}_initial_to_final.gif", initial_np, final_np, sizes, fixed,
                               benchmark.canvas_width, benchmark.canvas_height,
                               f"{benchmark_name}: initial.plc to final output")
        metrics = {"benchmark": benchmark_name, "initial": initial_metrics.__dict__, "final": final_metrics.__dict__}
        (out_dir / f"{benchmark_name}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    finally:
        os.chdir(original_cwd)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=repo_root() / "artifacts")
    parser.add_argument("--challenge-repo", type=Path, default=None)
    parser.add_argument("--benchmark", default="ibm01", choices=IBM_BENCHMARKS)
    parser.add_argument("--run-placer", action="store_true",
                        help="Run the placer on the selected benchmark and emit final placement/GIF artifacts.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(argv)
    ensure_out_dir(args.out_dir)
    save_static_score_artifacts(args.out_dir)
    if args.challenge_repo is not None:
        run_real_benchmark_artifacts(args.out_dir, args.challenge_repo.resolve(), args.benchmark, args.run_placer)
    print(f"Wrote artifacts to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
