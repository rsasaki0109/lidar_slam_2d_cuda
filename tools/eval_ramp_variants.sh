#!/bin/bash
# Evaluate all ramp range-weight variants against GT
set -e
cd lidar_slam_2d

GT="data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/ramp/ground_truth.tum"
CARTO="runs/iilabs_ramp_carto_at_slamx_s2k.csv"

echo "=== Baseline (no range weight) ==="
echo "--- slamx baseline ---"
env -u PYTHONPATH .venv/bin/slamx eval ate runs/iilabs_ramp_s2k_hybrid_refdense_cv_s2_refinegrid --gt "$GT" --max-dt-ms 50 2>/dev/null
echo "--- slamx baseline no-align ---"
env -u PYTHONPATH .venv/bin/slamx eval ate runs/iilabs_ramp_s2k_hybrid_refdense_cv_s2_refinegrid --gt "$GT" --max-dt-ms 50 --no-align 2>/dev/null

for variant in linear sigmoid combo; do
    run="runs/iilabs_ramp_s2k_rangeweight_${variant}"
    if [ -f "${run}/trajectory.json" ]; then
        echo ""
        echo "=== rangeweight_${variant} ==="
        echo "--- align ---"
        env -u PYTHONPATH .venv/bin/slamx eval ate "$run" --gt "$GT" --max-dt-ms 50 2>/dev/null
        echo "--- no-align ---"
        env -u PYTHONPATH .venv/bin/slamx eval ate "$run" --gt "$GT" --max-dt-ms 50 --no-align 2>/dev/null
    else
        echo ""
        echo "=== rangeweight_${variant} === (not yet complete)"
    fi
done

echo ""
echo "=== Cartographer sampled ==="
env -u PYTHONPATH .venv/bin/slamx eval ate "$CARTO" --gt "$GT" --max-dt-ms 50 2>/dev/null
echo "--- no-align ---"
env -u PYTHONPATH .venv/bin/slamx eval ate "$CARTO" --gt "$GT" --max-dt-ms 50 --no-align 2>/dev/null
