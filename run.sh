#!/usr/bin/env bash
set -euo pipefail

python scripts/run_par0029.py \
  --fixed /home/x_ma/test_data/generated_pairs/N21VMR_abd_001/extracted_vols/vol_t000.nii.gz \
  --moving /home/x_ma/test_data/generated_pairs/N21VMR_abd_001/extracted_vols/vol_t001.nii.gz \
  --out /home/x_ma/test_data/groupwise_processed