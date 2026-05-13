# Fast Iterative Registration of 3D Celtic Coins

Complete implementation of Generalized ICP for rigid registration of 3D point clouds using the Gauss-Newton algorithm with Lie algebra updates. I propose a fast GPU implementation in Metal and Objective-C++, inspired by VGICP. I applied the method to the Riedones3D coin benchmark.

<table>
  <tr>
    <td align="center">
      <img src="img/icp_vs_gicp_source01.png" width="360" height="240">
      <br>
      <em>Source (bottom) and target (top).</em>
    </td>
    <td align="center">
      <img src="img/icp_vs_gicp_res04.png" width="360" height="240">
      <br>
      <em>ICP in red vs GICP in green (2 its).</em>
    </td>
  </tr>
</table>

**Project Architecture**

| Area | Path | Role |
| --- | --- | --- |
| Core algorithms | `src/` | ICP and GICP implementations |
| Metal VGICP (Apple) | `src/gicp_metal.mm` + `metal/gicp_vgicp.metal` | GPU voxel-hash correspondences + GPU Hessian/gradient accumulation |
| Public headers | `include/` | Shared interfaces and math utilities |
| Coin benchmark | `examples/coin_benchmark.cpp` | Full Riedones3D benchmark (ICP vs GICP, SRE/SSE logging) |
| Coin demo | `examples/gicp_icp_coin.cpp` | Single-coin demo with controlled perturbations |
| Noise robustness | `examples/noise_robustness.cpp` | Jitter + rotation + translation sweep (RMSE logging) |
| Dataset | `Riedones3D_Registration_Benchmark/` | Coin point clouds and pair lists |
| Results | `logs/` | Run outputs (CSV + metadata) |
| Figures | `img/` | Report-ready images |

Before running the GICP experiments, download the Riedones3D dataset and place it in the repo root (`Riedones3D_Registration_Benchmark/`): https://cloud.minesparis.psl.eu/index.php/s/AVhOXp54IVGYLTM *(Segal et al., 2009; Horache et al., 2021)*.

**Build**

```bash
cmake -B build -S .
cmake --build build -j
```

**Run: Coin Benchmark (Riedones3D)**

```bash
./build/coin_benchmark \
  --dataset_root Riedones3D_Registration_Benchmark/test \
  --max_iter 20 \
  --k_neighbors 20 \
  --epsilon 1e-3 \
  --sample_ratio 1.0 \
  --log_dir logs
```

**Run: Single-Coin Demo**

```bash
./build/gicp_icp_coin \
  --input Riedones3D_Registration_Benchmark/test/die_obverse_0001/L0001D.ply \
  --angle_deg 10 \
  --trans_norm 0.01 \
  --noise_sigma 0.001 \
  --max_iter 20
```

**Run: Jitter Robustness Sweep**

```bash
./build/riedones3d_noise \
  --dataset_root Riedones3D_Registration_Benchmark/test \
  --num_pieces 5 \
  --jitter_list 0.001,0.01,0.1 \
  --rot_deg_list 50,100 \
  --trans_list 0,0.005,0.01 \
  --max_iter 20 \
  --log_dir logs
```

**Run: Metal VGICP Demo (Apple Silicon / macOS)**

```bash
./build/gicp_metal_demo
```

Useful options:

```bash
./build/gicp_metal_demo \
  --trials 10 \
  --points 12000 \
  --covariance_mode voxel \
  --downsample_resolution 0.0
```

This demo runs random SE(3) trials and reports:
- covariance build time (`cov_ms`)
- solve/iteration time (`solve_ms`)
- end-to-end time (`total_ms = cov_ms + solve_ms`)
- convergence success rate
- speedup CPU vs Metal

Covariance modes:
- `--covariance_mode knn`: faithful point-kNN covariance (more accurate, slower preprocessing)
- `--covariance_mode voxel`: fast voxel-approx covariance (much faster end-to-end)

Observed on this machine (12k points, 10 trials, `covariance_mode=voxel`, no downsample): solve stage ~`x84`, end-to-end ~`x29.6`, convergence `10/10`.

**Results and Outputs**

- Coin benchmark logs: `logs/coins_YYYYMMDD_HHMMSS/results.csv` and `logs/coins_YYYYMMDD_HHMMSS/summary.txt`
- Noise robustness logs: `logs/riedones3d_noise_YYYYMMDD_HHMMSS/results.csv`, `logs/riedones3d_noise_YYYYMMDD_HHMMSS/summary.csv`, and `logs/riedones3d_noise_YYYYMMDD_HHMMSS/metadata.txt`
