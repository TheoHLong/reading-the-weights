#!/usr/bin/env bash
# Wave 3 runner — paste-into-Colab friendly.
#
# Assumes:
#   - PWD is the project root (.../reading-the-weights)
#   - GPU runtime is selected
#   - One-time `bash scripts/setup_colab.sh` has already been run earlier
#     in the notebook to pip-install requirements + the project package
#
# What it runs (in order):
#   A. random MLP + noise, seeds 43 + 44   (≈ 30 min × 2 on a T4)
#   B. random MLP + noise, 4-on-5 mapping  (≈ 30 min)
#   C. early_stop@300 on trained CNN regime + on random CNN noise regime (≈ 30 min × 2)
#   D. measure_guide_rank.py over the 6 noise/trained guide combinations (a few min total)
#
# Total wall-clock on T4: ~3 hours. On A100: ~1 hour.
#
# Each training run streams its own metrics.csv + summary.json under
# results/metrics/<experiment_name>_<timestamp>/. Rank diagnostics land in
# results/diagnostics/guide_rank/.

set -euo pipefail

run() {
  local cfg="$1"
  echo
  echo "============================================================"
  echo " >> $cfg"
  echo "============================================================"
  python scripts/train_transfer.py --config "$cfg"
}

echo "[wave3] starting at $(date -u +%FT%TZ)"
echo "[wave3] CUDA visible: $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")' 2>/dev/null || echo unknown)"

# --- A. random MLP + noise, multi-seed -------------------------------------
run configs/transfer/cifar10_cka_n4_random_mlp_noise_s43.yaml
run configs/transfer/cifar10_cka_n4_random_mlp_noise_s44.yaml

# --- B. random MLP + noise, 4-on-5 mapping (confound control) --------------
run configs/transfer/cifar10_cka_n4_random_mlp_noise_mapped4.yaml

# --- C. early_stop on stable regimes ---------------------------------------
run configs/transfer/cifar10_cka_n4_earlystop300.yaml
run configs/transfer/cifar10_cka_n4_random_cnn_noise_earlystop300.yaml

# --- D. guide rank diagnostics ---------------------------------------------
echo
echo "============================================================"
echo " >> measure_guide_rank.py over wave-2/3 guide configs"
echo "============================================================"

python scripts/measure_guide_rank.py \
  --batch-size 256 --num-batches 4 \
  --config configs/transfer/cifar10_cka_n4.yaml                                   \
  --config configs/transfer/cifar10_cka_n4_s43.yaml                               \
  --config configs/transfer/cifar10_cka_n4_s44.yaml                               \
  --config configs/transfer/cifar10_cka_n4_random_cnn_noise.yaml                  \
  --config configs/transfer/cifar10_cka_n4_random_cnn_noise_s43.yaml              \
  --config configs/transfer/cifar10_cka_n4_random_cnn_noise_s44.yaml              \
  --config configs/transfer/cifar10_cka_n4_random_cnn.yaml                        \
  --config configs/transfer/cifar10_cka_n4_random_mlp.yaml                        \
  --config configs/transfer/cifar10_cka_n4_random_mlp_noise.yaml                  \
  --config configs/transfer/cifar10_cka_n4_random_mlp_noise_s43.yaml              \
  --config configs/transfer/cifar10_cka_n4_random_mlp_noise_s44.yaml              \
  --config configs/transfer/cifar10_cka_n4_random_mlp_noise_mapped4.yaml

echo
echo "[wave3] done at $(date -u +%FT%TZ)"
echo "[wave3] training metrics → results/metrics/"
echo "[wave3] guide rank      → results/diagnostics/guide_rank/"
