#!/usr/bin/env bash
# Reproduces every number and figure, in the order of LOG.md. About 30 min on 30 cores.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p data figures results logs

echo "== step 1: potential-flow coefficients (BEM) and checks"
$PY src/bem_two_barges.py --gaps 1.5 2 3 4 5 --res 1.0 --nw 60
$PY src/bem_two_barges.py --gaps 2 --res 0.5 --nw 60
$PY src/check_bem.py

echo "== step 4 fix: radiation damping 2.2-4 rad/s (needed by the time-domain model)"
$PY src/bem_high_freq.py

export OMP_NUM_THREADS=1   # parallel stages: one thread per worker
echo "== step 2: linear prediction in irregular seas"
$PY src/linear_fd.py
echo "== step 3: synthetic truth, gates"
$PY src/check_td.py
$PY src/diag_out_of_band.py
$PY src/check_hf_sensitivity.py
echo "== step 4: dataset"
$PY src/make_dataset.py --n 1200 --workers "${WORKERS:-30}"
$PY src/check_dataset.py
echo "== step 5: learning"
$PY src/train_compare.py --protocol A
$PY src/train_compare.py --protocol B
$PY src/summarise_step5.py results/step5A_results.csv
$PY src/summarise_step5.py results/step5B_results.csv
$PY src/plot_step5.py
$PY src/make_figures.py
$PY src/make_report.py
echo "done"
