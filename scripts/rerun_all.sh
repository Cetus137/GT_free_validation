#!/usr/bin/env bash
# Full recalibration pipeline. Usage: scripts/rerun_all.sh [SWITCH_CAP]
# SWITCH_CAP = max swap-partner distance in median spacings (1.1 = adjacent cells only).
# Writes results under data/_results/ with the standard names used by the figure scripts.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/Users/ewheeler/miniforge3/envs/valid_env/bin/python}
CAP=${1:-1.1}
R=data/_results
echo "=== switch cap ${CAP} spacings ==="

echo "--- 1. main sweep: six types, real + slab simulations"
$PY scripts/ce_error_sweep.py --sim --sim-shape slab --switch-cap "$CAP" --out $R/ce_error_sweep_slab_delaunay.csv > $R/log_slab.txt 2>&1
echo "--- 2. calibration sweep, four types, real only (outlier channels)"
$PY scripts/ce_error_sweep.py --types switch frag miss fp --switch-cap "$CAP" --out $R/ce_error_sweep_out_delaunay.csv > $R/log_out.txt 2>&1
echo "--- 3. low-rate sweep"
$PY scripts/ce_error_sweep.py --types switch frag miss fp --switch-cap "$CAP" --rates 0.0025 0.005 0.01 0.015 0.02 --out $R/ce_error_sweep_low_out_delaunay.csv > $R/log_low.txt 2>&1
echo "--- 4. switch geometry: slab and cube, switch only"
$PY scripts/ce_error_sweep.py --sim --sim-shape slab --types switch --switch-cap "$CAP" --out $R/ce_slab_switch.csv > $R/log_geo_slab.txt 2>&1
$PY scripts/ce_error_sweep.py --sim --sim-shape cube --types switch --switch-cap "$CAP" --out $R/ce_error_sweep_t1only_delaunay.csv > $R/log_geo_cube.txt 2>&1
echo "--- 5. frame-stride sweeps 2, 3, 4"
for s in 2 3 4; do
  $PY scripts/ce_error_sweep.py --frame-stride $s --stride 2 --types switch frag miss fp --switch-cap "$CAP" --out $R/ce_error_sweep_fs${s}_delaunay.csv > $R/log_fs${s}.txt 2>&1
done
echo "--- 6. estimator: mixtures, pooled, on-sweep"
$PY scripts/estimate_error_mix.py --sparse --constant --out $R/error_mix_validation.csv > $R/log_est.txt 2>&1
echo "--- 7. estimator: low rate with outlier channels"
$PY scripts/estimate_error_mix.py --outliers --outlier-calib $R/ce_error_sweep_low_out_delaunay.csv --sweep $R/ce_error_sweep_out_delaunay.csv --on-sweep $R/ce_error_sweep_low_out_delaunay.csv --skip-mixtures --constant --scenario-scale 0.25 --out $R/error_mix_low_out.csv > $R/log_est_low.txt 2>&1
echo "--- 8. cross-embryo, stride accuracy, division handling"
$PY scripts/cross_embryo.py > $R/log_cross.txt 2>&1
$PY scripts/stride_accuracy.py --pooled > $R/log_stride_acc.txt 2>&1
$PY scripts/ce_division_handling.py > $R/log_div.txt 2>&1
echo "--- 9. linker test with per-stride calibration"
$PY scripts/run_linker_test.py --frame-stride 1 2 3 4 --caps 1.0 --calib-pattern "$R/ce_error_sweep_fs{stride}_delaunay.csv" --out $R/linker_test_stride_calibrated.csv > $R/log_linker.txt 2>&1
echo "--- 10. figures"
$PY scripts/make_panels.py > $R/log_panels.txt 2>&1
$PY scripts/make_figure_ce_response.py > /dev/null 2>&1 || true
$PY scripts/make_figure_error_mix.py > /dev/null 2>&1 || true
$PY scripts/make_figure_switch_geometry.py > /dev/null 2>&1 || true
echo "=== done ==="
