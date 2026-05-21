"""
Soft-overlap simulated annealing macro placer.

The placer starts from the competition-provided ``initial.plc`` coordinates,
uses a soft overlap penalty to repair hard-macro legality with low displacement,
and then refines legal placements with an incremental proxy estimator. The
official challenge evaluator remains the source of truth for final scoring.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics
from submissions.retryoos.incremental_sa import run_incremental_sa
from submissions.retryoos.soft_overlap_sa import (
    run_soft_overlap_sa,
    _build_sep_tables,
    _total_overlap_area,
)
from submissions.retryoos.replace_sa import _load_plc


_CHUNK_SIZE = 200_000


def _build_numba_inputs(benchmark: Benchmark, pos_start: np.ndarray, gap: float):
    """Build all Numba-friendly arrays for the SA loops."""
    n_hard = benchmark.num_hard_macros
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    g_cols, g_rows = benchmark.grid_cols, benchmark.grid_rows
    bin_w = cw / g_cols; bin_h = ch / g_rows; bin_area = bin_w * bin_h
    hroutes = float(benchmark.hroutes_per_micron)
    vroutes = float(benchmark.vroutes_per_micron)

    sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    W = sizes_np[:, 0]; H = sizes_np[:, 1]
    half_w = W / 2; half_h = H / 2

    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_idx = np.where(movable)[0].astype(np.int32)

    port = benchmark.port_positions.numpy()
    all_pos = np.vstack([pos_start, port]) if len(port) > 0 else pos_start

    net_nodes_np = [n.numpy() for n in benchmark.net_nodes]
    net_w = benchmark.net_weights.numpy()
    num_nets = benchmark.num_nets

    fmin_x = np.full(num_nets, np.inf); fmax_x = np.full(num_nets, -np.inf)
    fmin_y = np.full(num_nets, np.inf); fmax_y = np.full(num_nets, -np.inf)
    net_offsets = [0]; net_pins = []
    macro_nets_list = [[] for _ in range(n_hard)]

    for i, nodes in enumerate(net_nodes_np):
        for nd in nodes:
            if nd < n_hard and movable[nd]:
                net_pins.append(nd)
                macro_nets_list[nd].append(i)
            else:
                px, py = all_pos[nd, 0], all_pos[nd, 1]
                if px < fmin_x[i]: fmin_x[i] = px
                if px > fmax_x[i]: fmax_x[i] = px
                if py < fmin_y[i]: fmin_y[i] = py
                if py > fmax_y[i]: fmax_y[i] = py
        net_offsets.append(len(net_pins))

    net_offsets = np.array(net_offsets, dtype=np.int32)
    net_pins = np.array(net_pins, dtype=np.int32)

    mto = np.zeros(n_hard + 1, dtype=np.int32)
    for m in range(n_hard):
        mto[m + 1] = mto[m] + len(macro_nets_list[m])
    mtl = np.empty(int(mto[-1]), dtype=np.int32)
    for m in range(n_hard):
        for j, k in enumerate(macro_nets_list[m]):
            mtl[mto[m] + j] = k

    static_density = np.zeros((g_cols, g_rows), dtype=np.float64)
    for i in range(benchmark.num_macros):
        if i < n_hard and movable[i]:
            continue
        px = pos_start[i, 0]; py = pos_start[i, 1]
        hw_i = float(benchmark.macro_sizes[i, 0]) / 2
        hh_i = float(benchmark.macro_sizes[i, 1]) / 2
        c1 = max(0, int((px - hw_i) / bin_w)); c2 = min(g_cols-1, int((px + hw_i) / bin_w))
        r1 = max(0, int((py - hh_i) / bin_h)); r2 = min(g_rows-1, int((py + hh_i) / bin_h))
        for c in range(c1, c2+1):
            bx0 = c*bin_w; bx1 = bx0+bin_w
            ox = max(0.0, min(px+hw_i, bx1)-max(px-hw_i, bx0))
            if ox <= 0: continue
            for r in range(r1, r2+1):
                by0 = r*bin_h; by1 = by0+bin_h
                oy = max(0.0, min(py+hh_i, by1)-max(py-hh_i, by0))
                if oy > 0: static_density[c, r] += ox * oy

    wl_norm = max(1.0, (cw + ch) * max(1, num_nets))
    const_hpwl = 0.0
    for i in range(num_nets):
        if net_offsets[i] == net_offsets[i+1] and fmin_x[i] != np.inf:
            const_hpwl += net_w[i] * ((fmax_x[i]-fmin_x[i]) + (fmax_y[i]-fmin_y[i]))

    return dict(
        n_hard=n_hard, cw=cw, ch=ch, W=W, H=H, half_w=half_w, half_h=half_h,
        movable_idx=movable_idx,
        num_nets=num_nets, net_offsets=net_offsets, net_pins=net_pins, net_w=net_w,
        fmin_x=fmin_x, fmax_x=fmax_x, fmin_y=fmin_y, fmax_y=fmax_y,
        macro_to_nets_offsets=mto, macro_to_nets_list_arr=mtl,
        static_density=static_density,
        g_cols=g_cols, g_rows=g_rows, bin_w=bin_w, bin_h=bin_h, bin_area=bin_area,
        hroutes=hroutes, vroutes=vroutes, wl_norm=wl_norm, const_hpwl=const_hpwl,
        gap=gap, movable_mask=movable,
    )


class SoftOverlapSAPlacer:
    """
    Phase 1: Soft-Overlap SA from initial.plc (resolves overlaps minimally).
    Phase 2: Hard-reject incremental SA from the quality-preserved start.
    """
    def __init__(self, seed: int = 42, gap: float = 0.005,
                 wall_target_small: float = 900.0,
                 wall_target_medium: float = 1200.0,
                 wall_target_large: float = 1800.0,
                 small_thresh: int = 300,
                 medium_thresh: int = 600,
                 soft_iters: int = 1_000_000):
        self.seed = seed
        self.gap = gap
        self.wall_target_small = wall_target_small
        self.wall_target_medium = wall_target_medium
        self.wall_target_large = wall_target_large
        self.small_thresh = small_thresh
        self.medium_thresh = medium_thresh
        self.soft_iters = soft_iters

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        n_hard = benchmark.num_hard_macros
        n_macro = benchmark.num_macros
        plc = _load_plc(benchmark.name)

        N = int((~benchmark.macro_fixed[:n_hard]).sum())
        if N < self.small_thresh:
            wall_target = self.wall_target_small
        elif N < self.medium_thresh:
            wall_target = self.wall_target_medium
        else:
            wall_target = self.wall_target_large

        # === Use initial.plc directly (NOT V46 baseline) ===
        pos_start = benchmark.macro_positions.numpy().copy()

        # Check initial quality
        t_init = torch.tensor(pos_start, dtype=torch.float32)
        ov0 = compute_overlap_metrics(t_init, benchmark)
        c0 = compute_proxy_cost(t_init, benchmark, plc) if plc else None
        init_proxy = c0["proxy_cost"] if c0 else 0
        init_overlaps = ov0["overlap_count"]
        print(f"  [SoftOverlapSA] N={N} K={benchmark.num_nets} wall={wall_target:.0f}s")
        print(f"  [SoftOverlapSA] initial.plc: proxy={init_proxy:.4f} overlaps={init_overlaps}")

        d = _build_numba_inputs(benchmark, pos_start, self.gap)

        # === Numba warmup ===
        warmup_pos = pos_start[:n_hard].copy()
        run_incremental_sa(
            warmup_pos, d["n_hard"], d["movable_idx"], d["cw"], d["ch"],
            d["W"], d["H"], d["half_w"], d["half_h"], d["gap"],
            d["num_nets"], d["net_offsets"], d["net_pins"], d["net_w"],
            d["fmin_x"], d["fmax_x"], d["fmin_y"], d["fmax_y"],
            d["macro_to_nets_offsets"], d["macro_to_nets_list_arr"],
            10, 1.0, 0.1, self.seed,
            d["static_density"].copy(), d["g_cols"], d["g_rows"],
            d["bin_w"], d["bin_h"], d["bin_area"],
            d["hroutes"], d["vroutes"], d["wl_norm"], d["const_hpwl"],
        )
        # Warmup soft overlap SA (to trigger JIT compilation)
        run_soft_overlap_sa(
            pos_start[:n_hard].copy(),
            d["n_hard"], d["movable_idx"], d["cw"], d["ch"],
            d["W"], d["H"], d["half_w"], d["half_h"], d["gap"],
            d["num_nets"], d["net_offsets"], d["net_pins"], d["net_w"],
            d["fmin_x"], d["fmax_x"], d["fmin_y"], d["fmax_y"],
            d["macro_to_nets_offsets"], d["macro_to_nets_list_arr"],
            d["static_density"].copy(),
            d["g_cols"], d["g_rows"], d["bin_w"], d["bin_h"], d["bin_area"],
            d["hroutes"], d["vroutes"], d["wl_norm"], d["const_hpwl"],
            n_iters_soft=10, lambda_ov_start=100.0, lambda_ov_end=100.0,
            init_temp=0.005, seed=self.seed,
        )

        # === Phase 1: Soft-Overlap SA Legalization ===
        if init_overlaps > 0:
            print(f"  [SoftOverlapSA] Phase 1: Soft-overlap SA ({self.soft_iters:,} iters) ...")
            t_p1 = time.time()

            # Calibrate lambda: start with proxy/overlap balance
            sep_x, sep_y = _build_sep_tables(d["n_hard"], d["half_w"], d["half_h"])
            init_ov_area = _total_overlap_area(pos_start[:n_hard], d["n_hard"], sep_x, sep_y)
            # lambda such that overlap term ≈ proxy term
            lambda_start = max(1.0, init_proxy / max(1e-9, init_ov_area) * 0.5)
            lambda_end = lambda_start * 1000.0  # ramp up hard

            pos_soft, remaining_ov, accepts_soft = run_soft_overlap_sa(
                pos_start[:n_hard].copy(),
                d["n_hard"], d["movable_idx"], d["cw"], d["ch"],
                d["W"], d["H"], d["half_w"], d["half_h"], d["gap"],
                d["num_nets"], d["net_offsets"], d["net_pins"], d["net_w"],
                d["fmin_x"], d["fmax_x"], d["fmin_y"], d["fmax_y"],
                d["macro_to_nets_offsets"], d["macro_to_nets_list_arr"],
                d["static_density"],
                d["g_cols"], d["g_rows"], d["bin_w"], d["bin_h"], d["bin_area"],
                d["hroutes"], d["vroutes"], d["wl_norm"], d["const_hpwl"],
                n_iters_soft=self.soft_iters,
                lambda_ov_start=lambda_start,
                lambda_ov_end=lambda_end,
                init_temp=init_proxy * 0.005,
                seed=self.seed,
            )

            pos_mid = pos_start.copy()
            pos_mid[:n_hard] = pos_soft
            t_mid = torch.tensor(pos_mid, dtype=torch.float32)
            ov_mid = compute_overlap_metrics(t_mid, benchmark)
            c_mid = compute_proxy_cost(t_mid, benchmark, plc) if plc else None
            mid_proxy = c_mid["proxy_cost"] if c_mid else 0
            mid_overlaps = ov_mid["overlap_count"]
            print(f"  [SoftOverlapSA] Phase 1 done in {time.time()-t_p1:.0f}s: "
                  f"proxy={mid_proxy:.4f} overlaps={mid_overlaps} accepts={accepts_soft:,}")

            # If still overlapping, fall back to V8 micro-legalize for remaining
            if mid_overlaps > 0:
                print(f"  [SoftOverlapSA] Still {mid_overlaps} overlaps; applying V8 micro-legalize...")
                from macro_place.sa_v49 import _micro_legalize
                pos_soft = _micro_legalize(pos_soft, d["movable_idx"], benchmark, self.gap)
                pos_mid[:n_hard] = pos_soft
                t_mid2 = torch.tensor(pos_mid, dtype=torch.float32)
                ov_mid2 = compute_overlap_metrics(t_mid2, benchmark)
                c_mid2 = compute_proxy_cost(t_mid2, benchmark, plc) if plc else None
                print(f"  [SoftOverlapSA] After micro-legalize: proxy={c_mid2['proxy_cost'] if c_mid2 else 0:.4f} "
                      f"overlaps={ov_mid2['overlap_count']}")

            sa_start_pos = pos_soft
        else:
            print(f"  [SoftOverlapSA] No overlaps in initial.plc; skipping Phase 1")
            sa_start_pos = pos_start[:n_hard].copy()

        # === Phase 2: Hard-reject incremental SA ===
        setup_elapsed = time.time() - t0
        sa_wall = max(60.0, wall_target - setup_elapsed)
        print(f"  [SoftOverlapSA] Phase 2: Hard-reject SA (wall_budget={sa_wall:.0f}s) ...")

        # Rebuild inputs using the soft-legalized positions
        pos_for_phase2 = pos_start.copy()
        pos_for_phase2[:n_hard] = sa_start_pos
        d2 = _build_numba_inputs(benchmark, pos_for_phase2, self.gap)

        pos = sa_start_pos.copy()
        best_pos = pos.copy()
        best_cost = float("inf")
        total_accepts = 0; total_iters = 0
        chunk_seed = self.seed + 1000
        sa_t0 = time.time()
        init_temp = 0.0001; final_temp = 1e-6

        while True:
            elapsed = time.time() - sa_t0
            if elapsed >= sa_wall and total_iters >= 500_000:
                break
            t_prog = min(1.0, elapsed / max(1.0, sa_wall))
            chunk_init_t = init_temp * ((final_temp / init_temp) ** t_prog)
            chunk_final_t = init_temp * ((final_temp / init_temp) ** min(1.0, t_prog + 0.1))

            chunk_pos, chunk_cost, a, ro, rm = run_incremental_sa(
                pos, d2["n_hard"], d2["movable_idx"], d2["cw"], d2["ch"],
                d2["W"], d2["H"], d2["half_w"], d2["half_h"], d2["gap"],
                d2["num_nets"], d2["net_offsets"], d2["net_pins"], d2["net_w"],
                d2["fmin_x"], d2["fmax_x"], d2["fmin_y"], d2["fmax_y"],
                d2["macro_to_nets_offsets"], d2["macro_to_nets_list_arr"],
                _CHUNK_SIZE, chunk_init_t, chunk_final_t, chunk_seed,
                d2["static_density"], d2["g_cols"], d2["g_rows"],
                d2["bin_w"], d2["bin_h"], d2["bin_area"],
                d2["hroutes"], d2["vroutes"], d2["wl_norm"], d2["const_hpwl"],
            )
            if chunk_cost < best_cost:
                best_cost = chunk_cost
                best_pos = chunk_pos.copy()
            pos[:] = chunk_pos
            total_accepts += a; total_iters += _CHUNK_SIZE
            chunk_seed += 1

        sa_elapsed = time.time() - sa_t0
        print(f"  [SoftOverlapSA] Phase 2 done: iters={total_iters:,} accepts={total_accepts:,} "
              f"t={sa_elapsed:.0f}s iters/s={total_iters/max(1,sa_elapsed):.0f}")

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)

        if plc is not None:
            c = compute_proxy_cost(full_pos, benchmark, plc)
            print(f"  [SoftOverlapSA] Done in {time.time()-t0:.0f}s. Final proxy={c['proxy_cost']:.4f}")

        return full_pos


if __name__ == "__main__":
    pass
