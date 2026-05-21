import sys, math, time
import torch
import numpy as np
import numba
from pathlib import Path
from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost

def _load_plc(name):
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    return None

@numba.njit(cache=True)
def get_net_bbox(net_id, pos, fmin_x, fmax_x, fmin_y, fmax_y, net_offsets, net_pins):
    start = net_offsets[net_id]
    end = net_offsets[net_id+1]
    nx_min = fmin_x[net_id]; nx_max = fmax_x[net_id]
    ny_min = fmin_y[net_id]; ny_max = fmax_y[net_id]
    for j in range(start, end):
        pin = net_pins[j]
        px = pos[pin, 0]; py = pos[pin, 1]
        if px < nx_min: nx_min = px
        if px > nx_max: nx_max = px
        if py < ny_min: ny_min = py
        if py > ny_max: ny_max = py
    return nx_min, nx_max, ny_min, ny_max

@numba.njit(cache=True)
def check_overlap_single(m_id, px, py, pos, n_hard, sep_x, sep_y, gap):
    for j in range(n_hard):
        if m_id == j: continue
        dx = abs(px - pos[j, 0])
        dy = abs(py - pos[j, 1])
        if dx < sep_x[m_id, j] + gap and dy < sep_y[m_id, j] + gap:
            return True
    return False

@numba.njit(cache=True)
def compute_true_proxy_cost(pos, W, H, N, num_nets, net_offsets, net_pins, net_weights,
                            fmin_x, fmax_x, fmin_y, fmax_y,
                            static_density, g_cols, g_rows, bin_w, bin_h, bin_area,
                            hroutes, vroutes, wl_norm, const_hpwl):
    hpwl = const_hpwl
    rudy_h = np.zeros((g_cols, g_rows), dtype=np.float64)
    rudy_v = np.zeros((g_cols, g_rows), dtype=np.float64)
    
    for i in range(num_nets):
        start = net_offsets[i]
        end = net_offsets[i+1]
        if start == end and fmin_x[i] == np.inf: continue
        
        nx_min = fmin_x[i]; nx_max = fmax_x[i]
        ny_min = fmin_y[i]; ny_max = fmax_y[i]
        
        for j in range(start, end):
            pin = net_pins[j]
            px = pos[pin, 0]; py = pos[pin, 1]
            if px < nx_min: nx_min = px
            if px > nx_max: nx_max = px
            if py < ny_min: ny_min = py
            if py > ny_max: ny_max = py
            
        n_hpwl = (nx_max - nx_min) + (ny_max - ny_min)
        hpwl += net_weights[i] * n_hpwl
        
        if n_hpwl > 0:
            c1 = max(0, int(nx_min / bin_w))
            c2 = min(g_cols - 1, int(nx_max / bin_w))
            r1 = max(0, int(ny_min / bin_h))
            r2 = min(g_rows - 1, int(ny_max / bin_h))
            
            h_d = net_weights[i] * (ny_max - ny_min) / n_hpwl
            v_d = net_weights[i] * (nx_max - nx_min) / n_hpwl
            
            for c in range(c1, c2 + 1):
                for r in range(r1, r2 + 1):
                    rudy_h[c, r] += h_d
                    rudy_v[c, r] += v_d
                    
    wl_cost = hpwl / wl_norm
    
    density = static_density.copy()
    for i in range(N):
        px = pos[i, 0]; py = pos[i, 1]
        hw = W[i] * 0.5; hh = H[i] * 0.5
        
        c1 = max(0, int((px - hw) / bin_w))
        c2 = min(g_cols - 1, int((px + hw) / bin_w))
        r1 = max(0, int((py - hh) / bin_h))
        r2 = min(g_rows - 1, int((py + hh) / bin_h))
        
        for c in range(c1, c2 + 1):
            bx0 = c * bin_w; bx1 = bx0 + bin_w
            ox = max(0.0, min(px + hw, bx1) - max(px - hw, bx0))
            if ox <= 0: continue
            for r in range(r1, r2 + 1):
                by0 = r * bin_h; by1 = by0 + bin_h
                oy = max(0.0, min(py + hh, by1) - max(py - hh, by0))
                if oy > 0:
                    density[c, r] += ox * oy
                    
    flat_den = (density / bin_area).flatten()
    flat_den = np.sort(flat_den)
    k10 = max(1, int(0.1 * len(flat_den)))
    den_cost = 0.5 * np.sum(flat_den[-k10:]) / k10
    
    hn = max(1e-9, bin_w * hroutes)
    vn = max(1e-9, bin_h * vroutes)
    h_norm = rudy_h / hn
    v_norm = rudy_v / vn
    comb = (h_norm + v_norm).flatten()
    comb = np.sort(comb)
    k5 = max(1, int(0.05 * len(comb)))
    cong_cost = 0.5 * np.sum(comb[-k5:]) / k5
    
    return wl_cost + den_cost + cong_cost

@numba.njit(cache=True)
def run_true_proxy_sa(pos, n_hard, movable_idx, cw, ch, W, H, half_w, half_h, gap,
                       num_nets, net_offsets, net_pins, net_weights,
                       fmin_x, fmax_x, fmin_y, fmax_y,
                       max_iters, init_temp, final_temp, seed,
                       static_density, g_cols, g_rows, bin_w, bin_h, bin_area, 
                       hroutes, vroutes, wl_norm, const_hpwl):
    np.random.seed(seed)
    
    sep_x = np.empty((n_hard, n_hard), dtype=np.float64)
    sep_y = np.empty((n_hard, n_hard), dtype=np.float64)
    for i in range(n_hard):
        for j in range(n_hard):
            sep_x[i, j] = half_w[i] + half_w[j]
            sep_y[i, j] = half_h[i] + half_h[j]

    cur_cost = compute_true_proxy_cost(pos, W, H, n_hard, num_nets, net_offsets, net_pins, net_weights,
                                       fmin_x, fmax_x, fmin_y, fmax_y,
                                       static_density, g_cols, g_rows, bin_w, bin_h, bin_area,
                                       hroutes, vroutes, wl_norm, const_hpwl)

    best_cost = cur_cost
    best_pos = pos.copy()
    
    accepts = 0
    rejects_overlap = 0
    num_movable = len(movable_idx)
    
    temp_factor = (final_temp / init_temp) ** (1.0 / max_iters)
    temp = init_temp

    for it in range(max_iters):
        m = movable_idx[np.random.randint(0, num_movable)]
        old_x = pos[m, 0]; old_y = pos[m, 1]
        
        # Micro shift only
        shift_mag = min(cw, ch) * 0.05 * (temp / init_temp)
        new_x = old_x + np.random.randn() * shift_mag
        new_y = old_y + np.random.randn() * shift_mag
        new_x = min(max(new_x, half_w[m] + gap), cw - half_w[m] - gap)
        new_y = min(max(new_y, half_h[m] + gap), ch - half_h[m] - gap)
        
        if check_overlap_single(m, new_x, new_y, pos, n_hard, sep_x, sep_y, gap):
            rejects_overlap += 1; temp *= temp_factor; continue

        pos[m, 0] = new_x; pos[m, 1] = new_y
        
        nc = compute_true_proxy_cost(pos, W, H, n_hard, num_nets, net_offsets, net_pins, net_weights,
                                       fmin_x, fmax_x, fmin_y, fmax_y,
                                       static_density, g_cols, g_rows, bin_w, bin_h, bin_area,
                                       hroutes, vroutes, wl_norm, const_hpwl)

        d = nc - cur_cost
        if d < 0 or np.random.random() < np.exp(-d / max(temp, 1e-12)):
            cur_cost = nc
            accepts += 1
            if nc < best_cost:
                best_cost = nc
                best_pos[:] = pos[:]
        else:
            pos[m, 0] = old_x; pos[m, 1] = old_y
            
        temp *= temp_factor
            
    return best_pos, accepts, rejects_overlap


class ReplaceRefinerPlacer:
    def __init__(self, seed=42, gap=0.005):
        self.seed = seed
        self.gap = gap

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
        g_cols, g_rows = benchmark.grid_cols, benchmark.grid_rows
        bin_w, bin_h = cw / g_cols, ch / g_rows
        bin_area = bin_w * bin_h
        
        sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        W = sizes_np[:, 0]
        H = sizes_np[:, 1]
        half_w = W / 2
        half_h = H / 2
        
        movable = benchmark.get_movable_mask()[:n_hard].numpy()
        movable_idx = np.where(movable)[0].astype(np.int32)
        N = len(movable_idx)
        if N == 0:
            return benchmark.macro_positions.clone()

        # Initialize and legalize baseline (RePlAce)
        from macro_place.sa_v49 import _legalize_v46
        print(f"  [Refiner] Legalizing RePlAce baseline...")
        pos_legal_hard, gap = _legalize_v46(benchmark)
        
        pos_legal = benchmark.macro_positions.numpy().copy()
        pos_legal[:n_hard] = pos_legal_hard
        
        plc = _load_plc(benchmark.name)
        if plc is not None:
            c = compute_proxy_cost(torch.tensor(pos_legal, dtype=torch.float32), benchmark, plc)
            print(f"  [Refiner] Legalized Baseline Proxy = {c['proxy_cost']:.4f}")

        # Phase 3: True Proxy Legal-Space Micro SA
        print(f"  [Refiner] Phase: True Proxy Legal-Space Micro SA...")
        iter_budget = 20_000_000 # Scaled up compute budget for Top 1

        init_pos = pos_legal.copy()
        port = benchmark.port_positions.numpy()
        all_pos = np.vstack([init_pos, port]) if len(port) > 0 else init_pos

        net_nodes_np = [n.numpy() for n in benchmark.net_nodes]
        net_w = benchmark.net_weights.numpy()
        num_nets = benchmark.num_nets
        
        fmin_x = np.full(num_nets, np.inf); fmax_x = np.full(num_nets, -np.inf)
        fmin_y = np.full(num_nets, np.inf); fmax_y = np.full(num_nets, -np.inf)
        
        net_offsets = [0]; net_pins = []

        for i, nodes in enumerate(net_nodes_np):
            for n in nodes:
                if n < n_hard and movable[n]:
                    net_pins.append(n)
                else:
                    px, py = all_pos[n, 0], all_pos[n, 1]
                    if px < fmin_x[i]: fmin_x[i] = px
                    if px > fmax_x[i]: fmax_x[i] = px
                    if py < fmin_y[i]: fmin_y[i] = py
                    if py > fmax_y[i]: fmax_y[i] = py
            net_offsets.append(len(net_pins))

        net_offsets = np.array(net_offsets, dtype=np.int32)
        net_pins = np.array(net_pins, dtype=np.int32)

        static_density = np.zeros((g_cols, g_rows), dtype=np.float64)
        for i in range(benchmark.num_macros):
            if i < n_hard and movable[i]: continue
            px = init_pos[i, 0]; py = init_pos[i, 1]
            hw_i = float(benchmark.macro_sizes[i, 0]) / 2; hh_i = float(benchmark.macro_sizes[i, 1]) / 2
            
            c1 = max(0, int((px - hw_i) / bin_w)); c2 = min(g_cols - 1, int((px + hw_i) / bin_w))
            r1 = max(0, int((py - hh_i) / bin_h)); r2 = min(g_rows - 1, int((py + hh_i) / bin_h))
            
            for c in range(c1, c2 + 1):
                bx0 = c * bin_w; bx1 = bx0 + bin_w
                ox = max(0.0, min(px + hw_i, bx1) - max(px - hw_i, bx0))
                if ox <= 0: continue
                for r in range(r1, r2 + 1):
                    by0 = r * bin_h; by1 = by0 + bin_h
                    oy = max(0.0, min(py + hh_i, by1) - max(py - hh_i, by0))
                    if oy > 0:
                        static_density[c, r] += ox * oy

        wl_norm = max(1.0, (cw + ch) * max(1, benchmark.num_nets))
        hroutes = float(benchmark.hroutes_per_micron)
        vroutes = float(benchmark.vroutes_per_micron)
        
        const_hpwl = 0.0
        for i in range(num_nets):
            if net_offsets[i] == net_offsets[i+1] and fmin_x[i] != np.inf:
                const_hpwl += net_w[i] * ((fmax_x[i] - fmin_x[i]) + (fmax_y[i] - fmin_y[i]))

        # Warmup
        run_true_proxy_sa(init_pos[:n_hard].copy(), n_hard, movable_idx, cw, ch, W, H, half_w, half_h, self.gap,
                           num_nets, net_offsets, net_pins, net_w,
                           fmin_x, fmax_x, fmin_y, fmax_y,
                           10, 1.0, 0.1, self.seed,
                           static_density.copy(), g_cols, g_rows, bin_w, bin_h, bin_area, 
                           hroutes, vroutes, wl_norm, const_hpwl)

        # Ultra-low temperature.
        init_temp = 0.0001
        final_temp = 0.000001
        
        pos = init_pos[:n_hard].copy()
        
        best_pos, accepts, rejects_ov = run_true_proxy_sa(
            pos, n_hard, movable_idx, cw, ch, W, H, half_w, half_h, self.gap,
            num_nets, net_offsets, net_pins, net_w,
            fmin_x, fmax_x, fmin_y, fmax_y,
            iter_budget, init_temp, final_temp, self.seed,
            static_density, g_cols, g_rows, bin_w, bin_h, bin_area, 
            hroutes, vroutes, wl_norm, const_hpwl
        )
        
        print(f"  [Refiner] SA Finished. Accepted: {accepts:,} | Rejected (Overlap): {rejects_ov:,}")
        
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)
        
        if plc is not None:
            c = compute_proxy_cost(full_pos, benchmark, plc)
            print(f"[Refiner] Done in {time.time()-t0:.1f}s. Final Proxy={c['proxy_cost']:.4f}")
            
        return full_pos

if __name__ == "__main__":
    pass
