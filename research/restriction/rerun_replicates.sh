#!/bin/sh
# Raise the principal resampling to 2,000 replicates. At a few hundred the endpoints of a
# 95% interval are the second or third most extreme draw, which is not an interval anyone
# should quote. Niced to the floor with one BLAS thread per job, so two cores in total.
#
#   nohup sh rerun_replicates.sh > /dev/null 2>&1 &
#
# Writes results/luce_null_2000.txt and results/sensitivity_2000.txt, leaving the 200-replicate
# files in place so the two can be compared.
cd "$(dirname "$0")" || exit 1
PY=../../.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "started $(date +%Y-%m-%dT%H:%M)" > results/REPLICATES_PROGRESS.txt
nice -n 19 $PY luce_null.py 2000 > results/luce_null_2000.txt 2>&1 &
A=$!
nice -n 19 $PY sensitivity.py 2000 > results/sensitivity_2000.txt 2>&1 &
B=$!
echo "luce_null pid $A, sensitivity pid $B" >> results/REPLICATES_PROGRESS.txt
wait $A; echo "luce_null exit $?" >> results/REPLICATES_PROGRESS.txt
wait $B; echo "sensitivity exit $?" >> results/REPLICATES_PROGRESS.txt
echo "finished $(date +%Y-%m-%dT%H:%M)" >> results/REPLICATES_PROGRESS.txt
