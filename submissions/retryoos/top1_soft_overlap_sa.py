"""
Soft-Overlap True-Proxy SA Legalization
=========================================
KEY INSIGHT: initial.plc IS the analytical (RePlAce) placement with small overlaps.
  ibm01: proxy=1.0385 with only 69 overlaps (RePlAce baseline = 0.9976)
  ibm09: proxy=1.1126 with 101 overlaps

Our V46 CD legalization DESTROYS this quality:
  ibm01: 1.0385 → 1.1808 = 0.18 proxy LOSS from legalization alone!

The fix: Soft-overlap SA that stays near initial.plc while resolving overlaps.
  Phase 1: cost = true_proxy + lambda * overlap_area_of_moved_macro
    → SA resolves overlaps via the overlap cost term
    → lambda ramps up to enforce legality
  Phase 2: hard-reject standard V55 SA (from near-initial.plc quality)
  → Expected ibm01: start at 1.04 → legalize to ~1.05 → SA to ~0.92-0.98

Python outer loop calling Numba inner kernels — correct design.
(Numba JIT cannot do Python imports inside the JIT function.)
"""
import math
import numpy as np

# Import Numba kernels from existing modules (static import only)
from submissions.retryoos.top1_incremental_sa import (
    _init_state,
    _compute_topk_cost,
    _apply_move,
)
import numba


@numba.njit(cache=True, fastmath=True)
def _macro_overlap_area(m, px, py, pos, n_hard, sep_x, sep_y):
    """Total overlap area between macro m at (px,py) and all other macros."""
    total = 0.0
    for j in range(n_hard):
        if m == j:
            continue
        dx = abs(px - pos[j, 0])
        dy = abs(py - pos[j, 1])
        ox = max(0.0, sep_x[m, j] - dx)
        oy = max(0.0, sep_y[m, j] - dy)
        total += ox * oy
    return total


@numba.njit(cache=True, fastmath=True)
def _total_overlap_area(pos, n_hard, sep_x, sep_y):
    """Total pairwise overlap area — O(N²)."""
    total = 0.0
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            dx = abs(pos[i, 0] - pos[j, 0])
            dy = abs(pos[i, 1] - pos[j, 1])
            ox = max(0.0, sep_x[i, j] - dx)
            oy = max(0.0, sep_y[i, j] - dy)
            total += ox * oy
    return total


@numba.njit(cache=True, fastmath=True)
def _build_sep_tables(n_hard, half_w, half_h):
    sep_x = np.empty((n_hard, n_hard), dtype=np.float64)
    sep_y = np.empty((n_hard, n_hard), dtype=np.float64)
    for i in range(n_hard):
        for j in range(n_hard):
            sep_x[i, j] = half_w[i] + half_w[j]
            sep_y[i, j] = half_h[i] + half_h[j]
    return sep_x, sep_y


def run_soft_overlap_sa(
    pos_in, n_hard, movable_idx, cw, ch,
    W, H, half_w, half_h, gap,
    num_nets, net_offsets, net_pins, net_weights,
    fmin_x, fmax_x, fmin_y, fmax_y,
    macro_to_nets_offsets, macro_to_nets_list,
    static_density, g_cols, g_rows, bin_w, bin_h, bin_area,
    hroutes, vroutes, wl_norm, const_hpwl,
    n_iters_soft=500_000,
    lambda_ov_start=100.0,
    lambda_ov_end=100_000.0,
    init_temp=0.005,
    seed=42,
):
    """
    Python loop calling Numba kernels for the soft-overlap legalization phase.
    Resolves overlaps in initial.plc while minimally moving macros.

    Returns: (best_pos, best_total_overlap_area, accepts)
    """
    rng = np.random.default_rng(seed)
    pos = pos_in.copy()

    sep_x, sep_y = _build_sep_tables(n_hard, half_w, half_h)

    # Initialize incremental state
    density, rudy_h, rudy_v, net_bbox, net_hpwl, total_hpwl = _init_state(
        pos, W, H, n_hard, num_nets,
        net_offsets, net_pins, net_weights,
        fmin_x, fmax_x, fmin_y, fmax_y,
        static_density, g_cols, g_rows, bin_w, bin_h,
    )

    proxy_cost = _compute_topk_cost(
        density, rudy_h, rudy_v, g_cols, g_rows, bin_area,
        hroutes, vroutes, bin_w, bin_h,
        total_hpwl, const_hpwl, wl_norm,
    )
    overlap_area = _total_overlap_area(pos, n_hard, sep_x, sep_y)
    lam = lambda_ov_start
    cur_cost = proxy_cost + lam * overlap_area

    best_pos = pos.copy()
    best_overlap = overlap_area
    best_cost = cur_cost

    num_movable = len(movable_idx)
    final_temp = init_temp * 1e-5
    temp_factor = (final_temp / init_temp) ** (1.0 / max(1, n_iters_soft))
    lam_factor = (lambda_ov_end / lambda_ov_start) ** (1.0 / max(1, n_iters_soft))
    temp = init_temp

    accepts = 0

    for it in range(n_iters_soft):
        m = int(movable_idx[rng.integers(0, num_movable)])
        old_x = float(pos[m, 0])
        old_y = float(pos[m, 1])

        # Small moves — preserve quality, just nudge overlapping macros apart
        shift_mag = min(cw, ch) * 0.02
        new_x = float(np.clip(old_x + rng.standard_normal() * shift_mag,
                               half_w[m] + gap, cw - half_w[m] - gap))
        new_y = float(np.clip(old_y + rng.standard_normal() * shift_mag,
                               half_h[m] + gap, ch - half_h[m] - gap))

        # Overlap delta
        old_ov_m = _macro_overlap_area(m, old_x, old_y, pos, n_hard, sep_x, sep_y)
        new_ov_m = _macro_overlap_area(m, new_x, new_y, pos, n_hard, sep_x, sep_y)
        d_overlap = new_ov_m - old_ov_m

        # Apply move for proxy delta (incremental)
        d_hpwl = _apply_move(
            m, new_x, new_y, pos, W, H,
            density, rudy_h, rudy_v, net_bbox, net_hpwl,
            macro_to_nets_offsets, macro_to_nets_list,
            net_offsets, net_pins, net_weights,
            fmin_x, fmax_x, fmin_y, fmax_y,
            bin_w, bin_h, g_cols, g_rows,
        )
        total_hpwl += d_hpwl

        new_proxy = _compute_topk_cost(
            density, rudy_h, rudy_v, g_cols, g_rows, bin_area,
            hroutes, vroutes, bin_w, bin_h,
            total_hpwl, const_hpwl, wl_norm,
        )

        new_cost = new_proxy + lam * (overlap_area + d_overlap)
        d_cost = new_cost - cur_cost

        if d_cost < 0.0 or rng.random() < math.exp(-d_cost / max(temp, 1e-12)):
            cur_cost = new_cost
            proxy_cost = new_proxy
            overlap_area += d_overlap
            accepts += 1
            if new_cost < best_cost:
                best_cost = new_cost
                best_overlap = overlap_area
                best_pos = pos.copy()
        else:
            # Unmake move
            d_back = _apply_move(
                m, old_x, old_y, pos, W, H,
                density, rudy_h, rudy_v, net_bbox, net_hpwl,
                macro_to_nets_offsets, macro_to_nets_list,
                net_offsets, net_pins, net_weights,
                fmin_x, fmax_x, fmin_y, fmax_y,
                bin_w, bin_h, g_cols, g_rows,
            )
            total_hpwl += d_back

        temp *= temp_factor
        lam *= lam_factor

    return best_pos, float(best_overlap), accepts
