"""
Incremental proxy simulated annealing (Numba JIT).

The slow reference SA in `top1_replace_sa.py` recomputes the entire net bbox
sweep, RUDY grid, density grid and sort on every iteration. This module keeps
that state incrementally so the inner loop can explore many more legal moves.

This module maintains the density / RUDY / net-bbox state and updates it
INCREMENTALLY on each move (touching only nets containing the moved macro):
  - density: O(bbox_bins_for_macro)
  - net bbox: O(degree(m) * pins_per_net)
  - RUDY: O(degree(m) * bbox_bins_per_net)
  - top-K cost: still O(G²) but with np.partition (linear) not np.sort.

Symmetric make/unmake pattern: same function applied with reverse args undoes.

IMPORTANT: this module uses @numba.njit(cache=True). DO NOT load it via
importlib — static `from submissions.retryoos.top1_incremental_sa import ...`
only, to avoid cache corruption.
"""
import numpy as np
import numba


@numba.njit(cache=True, fastmath=True)
def _density_delta(density, px, py, hw, hh, sign,
                   bin_w, bin_h, g_cols, g_rows):
    """Add sign × macro footprint to the density grid (sign=+1 or -1)."""
    c1 = max(0, int((px - hw) / bin_w))
    c2 = min(g_cols - 1, int((px + hw) / bin_w))
    r1 = max(0, int((py - hh) / bin_h))
    r2 = min(g_rows - 1, int((py + hh) / bin_h))
    for c in range(c1, c2 + 1):
        bx0 = c * bin_w
        bx1 = bx0 + bin_w
        ox = max(0.0, min(px + hw, bx1) - max(px - hw, bx0))
        if ox <= 0.0:
            continue
        for r in range(r1, r2 + 1):
            by0 = r * bin_h
            by1 = by0 + bin_h
            oy = max(0.0, min(py + hh, by1) - max(py - hh, by0))
            if oy > 0.0:
                density[c, r] += sign * ox * oy


@numba.njit(cache=True, fastmath=True)
def _rudy_delta(rudy_h, rudy_v, xl, xr, yl, yr, weight, sign,
                bin_w, bin_h, g_cols, g_rows):
    """Add sign × net's RUDY contribution given its bbox (xl,xr,yl,yr)."""
    span_x = xr - xl
    span_y = yr - yl
    hpwl = span_x + span_y
    if hpwl <= 0.0:
        return
    c1 = max(0, int(xl / bin_w))
    c2 = min(g_cols - 1, int(xr / bin_w))
    r1 = max(0, int(yl / bin_h))
    r2 = min(g_rows - 1, int(yr / bin_h))
    h_d = sign * weight * span_y / hpwl
    v_d = sign * weight * span_x / hpwl
    for c in range(c1, c2 + 1):
        for r in range(r1, r2 + 1):
            rudy_h[c, r] += h_d
            rudy_v[c, r] += v_d


@numba.njit(cache=True, fastmath=True)
def _recompute_net_bbox(k, pos, fmin_x, fmax_x, fmin_y, fmax_y,
                         net_offsets, net_pins):
    """Recompute bbox for one net from scratch."""
    start = net_offsets[k]
    end = net_offsets[k + 1]
    xmin = fmin_x[k]; xmax = fmax_x[k]
    ymin = fmin_y[k]; ymax = fmax_y[k]
    for j in range(start, end):
        pin = net_pins[j]
        px = pos[pin, 0]
        py = pos[pin, 1]
        if px < xmin: xmin = px
        if px > xmax: xmax = px
        if py < ymin: ymin = py
        if py > ymax: ymax = py
    return xmin, xmax, ymin, ymax


@numba.njit(cache=True, fastmath=True)
def _init_state(pos, W, H, n_hard, num_nets,
                net_offsets, net_pins, net_weights,
                fmin_x, fmax_x, fmin_y, fmax_y,
                static_density, g_cols, g_rows, bin_w, bin_h):
    """Build initial density, rudy, net_bbox, net_hpwl from positions."""
    density = static_density.copy()
    rudy_h = np.zeros((g_cols, g_rows), dtype=np.float64)
    rudy_v = np.zeros((g_cols, g_rows), dtype=np.float64)
    net_bbox = np.zeros((num_nets, 4), dtype=np.float64)
    net_hpwl = np.zeros(num_nets, dtype=np.float64)
    total_hpwl = 0.0

    # Density from all movable hard macros
    for i in range(n_hard):
        _density_delta(density, pos[i, 0], pos[i, 1],
                        W[i] * 0.5, H[i] * 0.5, +1.0,
                        bin_w, bin_h, g_cols, g_rows)

    # Per-net bbox + RUDY
    for k in range(num_nets):
        xmin, xmax, ymin, ymax = _recompute_net_bbox(
            k, pos, fmin_x, fmax_x, fmin_y, fmax_y, net_offsets, net_pins)
        net_bbox[k, 0] = xmin
        net_bbox[k, 1] = xmax
        net_bbox[k, 2] = ymin
        net_bbox[k, 3] = ymax
        hpwl_k = (xmax - xmin) + (ymax - ymin)
        net_hpwl[k] = hpwl_k * net_weights[k]
        total_hpwl += net_hpwl[k]
        # Skip nets with no pins and no fixed contributions
        if net_offsets[k] == net_offsets[k + 1] and fmin_x[k] == np.inf:
            continue
        _rudy_delta(rudy_h, rudy_v, xmin, xmax, ymin, ymax,
                     net_weights[k], +1.0, bin_w, bin_h, g_cols, g_rows)

    return density, rudy_h, rudy_v, net_bbox, net_hpwl, total_hpwl


@numba.njit(cache=True, fastmath=True)
def _compute_topk_cost(density, rudy_h, rudy_v,
                       g_cols, g_rows, bin_area,
                       hroutes, vroutes, bin_w, bin_h,
                       total_hpwl, const_hpwl, wl_norm):
    """Compute the incremental proxy estimate from current state via top-K partition."""
    n_bins = g_cols * g_rows

    # Flatten density / bin_area
    flat_d = (density / bin_area).flatten()
    k10 = max(1, int(0.1 * n_bins))
    # Use partition (O(n)) — Numba supports np.partition
    p_d = np.partition(flat_d, n_bins - k10)
    s_d = 0.0
    for i in range(n_bins - k10, n_bins):
        s_d += p_d[i]
    den_cost = 0.5 * s_d / k10

    hn = max(1e-9, bin_w * hroutes)
    vn = max(1e-9, bin_h * vroutes)
    comb = (rudy_h / hn + rudy_v / vn).flatten()
    k5 = max(1, int(0.05 * n_bins))
    p_c = np.partition(comb, n_bins - k5)
    s_c = 0.0
    for i in range(n_bins - k5, n_bins):
        s_c += p_c[i]
    cong_cost = 0.5 * s_c / k5

    wl_cost = (total_hpwl + const_hpwl) / wl_norm
    return wl_cost + den_cost + cong_cost


@numba.njit(cache=True, fastmath=True)
def _apply_move(m, nx, ny, pos, W, H,
                density, rudy_h, rudy_v, net_bbox, net_hpwl,
                macro_to_nets_offsets, macro_to_nets_list,
                net_offsets, net_pins, net_weights,
                fmin_x, fmax_x, fmin_y, fmax_y,
                bin_w, bin_h, g_cols, g_rows):
    """
    Apply move m → (nx, ny). Returns hpwl_delta (added to total_hpwl).
    Updates density, rudy_h, rudy_v, net_bbox, net_hpwl in-place.
    """
    ox = pos[m, 0]; oy = pos[m, 1]
    hw = W[m] * 0.5; hh = H[m] * 0.5

    # Density: remove old footprint, add new
    _density_delta(density, ox, oy, hw, hh, -1.0, bin_w, bin_h, g_cols, g_rows)
    _density_delta(density, nx, ny, hw, hh, +1.0, bin_w, bin_h, g_cols, g_rows)

    # Move the macro
    pos[m, 0] = nx
    pos[m, 1] = ny

    # Update each affected net: remove old RUDY, recompute bbox, add new RUDY
    start = macro_to_nets_offsets[m]
    end = macro_to_nets_offsets[m + 1]
    hpwl_delta = 0.0
    for j in range(start, end):
        k = macro_to_nets_list[j]
        # Subtract old RUDY
        _rudy_delta(rudy_h, rudy_v,
                     net_bbox[k, 0], net_bbox[k, 1],
                     net_bbox[k, 2], net_bbox[k, 3],
                     net_weights[k], -1.0, bin_w, bin_h, g_cols, g_rows)
        # Recompute bbox
        nxmin, nxmax, nymin, nymax = _recompute_net_bbox(
            k, pos, fmin_x, fmax_x, fmin_y, fmax_y, net_offsets, net_pins)
        new_hpwl = ((nxmax - nxmin) + (nymax - nymin)) * net_weights[k]
        hpwl_delta += new_hpwl - net_hpwl[k]
        # Commit
        net_bbox[k, 0] = nxmin
        net_bbox[k, 1] = nxmax
        net_bbox[k, 2] = nymin
        net_bbox[k, 3] = nymax
        net_hpwl[k] = new_hpwl
        # Add new RUDY
        _rudy_delta(rudy_h, rudy_v, nxmin, nxmax, nymin, nymax,
                     net_weights[k], +1.0, bin_w, bin_h, g_cols, g_rows)

    return hpwl_delta


@numba.njit(cache=True, fastmath=True)
def _check_overlap_single(m, px, py, pos, n_hard, sep_x, sep_y, gap):
    for j in range(n_hard):
        if m == j:
            continue
        dx = abs(px - pos[j, 0])
        dy = abs(py - pos[j, 1])
        if dx < sep_x[m, j] + gap and dy < sep_y[m, j] + gap:
            return True
    return False


@numba.njit(cache=True, fastmath=True)
def run_incremental_sa(
    pos, n_hard, movable_idx, cw, ch,
    W, H, half_w, half_h, gap,
    num_nets, net_offsets, net_pins, net_weights,
    fmin_x, fmax_x, fmin_y, fmax_y,
    macro_to_nets_offsets, macro_to_nets_list,
    max_iters, init_temp, final_temp, seed,
    static_density, g_cols, g_rows, bin_w, bin_h, bin_area,
    hroutes, vroutes, wl_norm, const_hpwl,
):
    """
    Incremental proxy SA with local state updates for faster per-iteration cost.
    """
    np.random.seed(seed)

    # Macro-macro separation tables
    sep_x = np.empty((n_hard, n_hard), dtype=np.float64)
    sep_y = np.empty((n_hard, n_hard), dtype=np.float64)
    for i in range(n_hard):
        for j in range(n_hard):
            sep_x[i, j] = half_w[i] + half_w[j]
            sep_y[i, j] = half_h[i] + half_h[j]

    # Initialize state
    density, rudy_h, rudy_v, net_bbox, net_hpwl, total_hpwl = _init_state(
        pos, W, H, n_hard, num_nets,
        net_offsets, net_pins, net_weights,
        fmin_x, fmax_x, fmin_y, fmax_y,
        static_density, g_cols, g_rows, bin_w, bin_h,
    )

    cur_cost = _compute_topk_cost(
        density, rudy_h, rudy_v, g_cols, g_rows, bin_area,
        hroutes, vroutes, bin_w, bin_h,
        total_hpwl, const_hpwl, wl_norm,
    )
    best_cost = cur_cost
    best_pos = pos.copy()

    accepts = 0
    rejects_overlap = 0
    rejects_metro = 0
    num_movable = len(movable_idx)

    temp_factor = (final_temp / init_temp) ** (1.0 / max(1, max_iters))
    temp = init_temp

    for it in range(max_iters):
        move_type = np.random.random()
        
        if move_type < 0.2 and num_movable > 1:
            m1 = movable_idx[np.random.randint(0, num_movable)]
            m2 = movable_idx[np.random.randint(0, num_movable)]
            while m1 == m2:
                m2 = movable_idx[np.random.randint(0, num_movable)]
                
            old_x1 = pos[m1, 0]; old_y1 = pos[m1, 1]
            old_x2 = pos[m2, 0]; old_y2 = pos[m2, 1]
            
            new_x1 = min(max(old_x2, half_w[m1] + gap), cw - half_w[m1] - gap)
            new_y1 = min(max(old_y2, half_h[m1] + gap), ch - half_h[m1] - gap)
            new_x2 = min(max(old_x1, half_w[m2] + gap), cw - half_w[m2] - gap)
            new_y2 = min(max(old_y1, half_h[m2] + gap), ch - half_h[m2] - gap)
            
            overlap = False
            for j in range(n_hard):
                if j != m1 and j != m2:
                    dx1 = abs(new_x1 - pos[j, 0])
                    dy1 = abs(new_y1 - pos[j, 1])
                    if dx1 < sep_x[m1, j] + gap and dy1 < sep_y[m1, j] + gap:
                        overlap = True; break
                    dx2 = abs(new_x2 - pos[j, 0])
                    dy2 = abs(new_y2 - pos[j, 1])
                    if dx2 < sep_x[m2, j] + gap and dy2 < sep_y[m2, j] + gap:
                        overlap = True; break
            if not overlap:
                dx = abs(new_x1 - new_x2)
                dy = abs(new_y1 - new_y2)
                if dx < sep_x[m1, m2] + gap and dy < sep_y[m1, m2] + gap:
                    overlap = True
                    
            if overlap:
                rejects_overlap += 1
                temp *= temp_factor
                continue
                
            d_hpwl1 = _apply_move(m1, new_x1, new_y1, pos, W, H, density, rudy_h, rudy_v, net_bbox, net_hpwl, macro_to_nets_offsets, macro_to_nets_list, net_offsets, net_pins, net_weights, fmin_x, fmax_x, fmin_y, fmax_y, bin_w, bin_h, g_cols, g_rows)
            d_hpwl2 = _apply_move(m2, new_x2, new_y2, pos, W, H, density, rudy_h, rudy_v, net_bbox, net_hpwl, macro_to_nets_offsets, macro_to_nets_list, net_offsets, net_pins, net_weights, fmin_x, fmax_x, fmin_y, fmax_y, bin_w, bin_h, g_cols, g_rows)
            total_hpwl += d_hpwl1 + d_hpwl2
            
            new_cost = _compute_topk_cost(density, rudy_h, rudy_v, g_cols, g_rows, bin_area, hroutes, vroutes, bin_w, bin_h, total_hpwl, const_hpwl, wl_norm)
            
            d_cost = new_cost - cur_cost
            if d_cost < 0.0 or np.random.random() < np.exp(-d_cost / max(temp, 1e-12)):
                cur_cost = new_cost
                accepts += 1
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_pos[:] = pos[:]
            else:
                db2 = _apply_move(m2, old_x2, old_y2, pos, W, H, density, rudy_h, rudy_v, net_bbox, net_hpwl, macro_to_nets_offsets, macro_to_nets_list, net_offsets, net_pins, net_weights, fmin_x, fmax_x, fmin_y, fmax_y, bin_w, bin_h, g_cols, g_rows)
                db1 = _apply_move(m1, old_x1, old_y1, pos, W, H, density, rudy_h, rudy_v, net_bbox, net_hpwl, macro_to_nets_offsets, macro_to_nets_list, net_offsets, net_pins, net_weights, fmin_x, fmax_x, fmin_y, fmax_y, bin_w, bin_h, g_cols, g_rows)
                total_hpwl += db2 + db1
                rejects_metro += 1
                
        else:
            m = movable_idx[np.random.randint(0, num_movable)]
            old_x = pos[m, 0]
            old_y = pos[m, 1]
    
            shift_mag = min(cw, ch) * 0.05 * (temp / init_temp)
            new_x = old_x + np.random.randn() * shift_mag
            new_y = old_y + np.random.randn() * shift_mag
            new_x = min(max(new_x, half_w[m] + gap), cw - half_w[m] - gap)
            new_y = min(max(new_y, half_h[m] + gap), ch - half_h[m] - gap)
    
            if _check_overlap_single(m, new_x, new_y, pos, n_hard, sep_x, sep_y, gap):
                rejects_overlap += 1
                temp *= temp_factor
                continue
    
            # Apply move incrementally
            d_hpwl = _apply_move(
                m, new_x, new_y, pos, W, H,
                density, rudy_h, rudy_v, net_bbox, net_hpwl,
                macro_to_nets_offsets, macro_to_nets_list,
                net_offsets, net_pins, net_weights,
                fmin_x, fmax_x, fmin_y, fmax_y,
                bin_w, bin_h, g_cols, g_rows,
            )
            total_hpwl += d_hpwl
    
            new_cost = _compute_topk_cost(
                density, rudy_h, rudy_v, g_cols, g_rows, bin_area,
                hroutes, vroutes, bin_w, bin_h,
                total_hpwl, const_hpwl, wl_norm,
            )
    
            d_cost = new_cost - cur_cost
            if d_cost < 0.0 or np.random.random() < np.exp(-d_cost / max(temp, 1e-12)):
                # Accept
                cur_cost = new_cost
                accepts += 1
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_pos[:] = pos[:]
            else:
                # Reject: unmake by applying the inverse move
                d_hpwl_back = _apply_move(
                    m, old_x, old_y, pos, W, H,
                    density, rudy_h, rudy_v, net_bbox, net_hpwl,
                    macro_to_nets_offsets, macro_to_nets_list,
                    net_offsets, net_pins, net_weights,
                    fmin_x, fmax_x, fmin_y, fmax_y,
                    bin_w, bin_h, g_cols, g_rows,
                )
                total_hpwl += d_hpwl_back
                rejects_metro += 1

        temp *= temp_factor

    return best_pos, best_cost, accepts, rejects_overlap, rejects_metro
