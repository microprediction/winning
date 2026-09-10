#!/bin/sh
# Overnight rerun of the two expensive tables under the restrictions-only estimand,
# 2 <= |T| < K. Niced to the floor and held to two cores, one per job, so the machine
# stays usable. Roughly five hours each, run in parallel, so about five hours total.
#
#   sh rerun_estimand.sh
#
# Writes results/luce_null_restrictions.txt and results/sensitivity_restrictions.txt.
# Neither overwrites a published file; the comparison against the old estimand is the
# point, so both versions have to survive.
cd "$(dirname "$0")" || exit 1
PY=../../.venv/bin/python
STAMP=$(date +%Y-%m-%dT%H:%M)

# one BLAS thread per job. Without this numpy grabs every core for matrices far too
# small to benefit, and the two jobs fight each other.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "started $STAMP" > results/RERUN_PROGRESS.txt

nice -n 19 $PY luce_null.py 200 > results/luce_null_restrictions.txt 2>&1 &
NULL_PID=$!
nice -n 19 $PY sensitivity.py 200 > results/sensitivity_restrictions.txt 2>&1 &
SENS_PID=$!

echo "luce_null pid $NULL_PID, sensitivity pid $SENS_PID" >> results/RERUN_PROGRESS.txt
wait $NULL_PID; echo "luce_null exit $?" >> results/RERUN_PROGRESS.txt
wait $SENS_PID; echo "sensitivity exit $?" >> results/RERUN_PROGRESS.txt
echo "finished $(date +%Y-%m-%dT%H:%M)" >> results/RERUN_PROGRESS.txt
