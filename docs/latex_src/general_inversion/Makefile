# Regenerate the numbers in "Scalable Inversion of Contests with Correlated
# Performances, Including Softmax and Multinomial Probit".
#
#   make env       build the pinned environment (.venv at the repository root)
#   make           the accuracy claims, about 15 minutes on one laptop core set
#   make full      everything else that is not a wall-clock figure (hours)
#   make timings   every wall-clock figure; refuses a busy machine (about 50 min)
#   make referee-r rerun the two R referees (needs R with mvtnorm,
#                  TruncatedNormal, jsonlite; see REPRODUCE.md)
#
# Every target writes its log to out/<name>.log. CLAIMS.md maps each number
# in the paper to the target and log line that produces it.

ROOT    := $(abspath ../../..)
PY      ?= $(ROOT)/.venv/bin/python
RSCRIPT ?= Rscript
OUT     := $(CURDIR)/out
THREADS := OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg

# run <name> <dir> <command...>: run in <dir>, log to out/<name>.log, fail loudly
define run
	@mkdir -p $(OUT)
	@echo "== $(1)"
	@cd $(2) && $(THREADS) $(3) > $(OUT)/$(1).log 2>&1 || { tail -20 $(OUT)/$(1).log; exit 1; }
	@grep -vE 'Warning|forward_grid|^ +x, points|^ +pb2? = ' $(OUT)/$(1).log | tail -6
endef

QUICK := tests stress referee ghk alt mislocate invert bm genz200 window tree \
         gh-vs-sobol split-referee multistart fuzz fixed-rule
FULL  := referee-r botev-tree ensembles kernel estimation covariates independent-inversion \
         large-n newton-krylov blp-ablation factor-rqmc law scale tenmillion invertmillion

.PHONY: all quick full timings env clean $(QUICK) $(FULL)

all quick: $(QUICK)
full: quick $(FULL)

env:
	cd $(ROOT) && uv venv -p 3.12 .venv && \
	  VIRTUAL_ENV=.venv uv pip install -r $(CURDIR)/requirements.lock && \
	  VIRTUAL_ENV=.venv uv pip install --no-deps -e .

# ---- quick: accuracy claims ------------------------------------------------
tests:
	$(call run,tests,$(ROOT),$(PY) -m pytest -q -p no:cacheprovider tests/test_race_invariants.py tests/test_sharp_factor_escalation.py tests/test_blocks.py)
stress:
	$(call run,stress,$(CURDIR),$(PY) break.py)
referee:   # Python side; replays the committed R output (make referee-r reruns R)
	$(call run,referee,$(CURDIR),$(PY) referee_cases.py && $(PY) referee_check.py)
ghk:
	$(call run,ghk,$(CURDIR),$(PY) bench.py ghk)
alt:
	$(call run,alt,$(CURDIR),$(PY) bench.py alt)
mislocate:
	$(call run,mislocate,$(CURDIR),$(PY) bench.py mislocate)
invert:
	$(call run,invert,$(CURDIR),$(PY) bench.py invert)
bm:
	$(call run,bm,$(CURDIR),$(PY) bench.py bm)
genz200:
	$(call run,genz200,$(CURDIR),$(PY) bench.py genz200)
window:
	$(call run,window,$(ROOT)/research/lattice_window,$(PY) run_window_savings.py)
tree:
	$(call run,tree,$(ROOT)/research/tree_exactness,$(PY) run_exactness.py)
gh-vs-sobol:
	$(call run,gh-vs-sobol,$(ROOT)/research/factor_nodes,$(PY) run_gh_vs_sobol.py)
split-referee:
	$(call run,split-referee,$(ROOT)/research/general_sigma,$(PY) run_split_referee.py --n 2000 --draws 2000000)
multistart:
	$(call run,multistart,$(ROOT)/research/general_sigma,$(PY) run_multistart.py --starts 8 --n 300)
fuzz:
	$(call run,fuzz,$(ROOT)/research/fuzz,$(PY) fuzz_races.py)

fixed-rule:
	$(call run,fixed-rule,$(ROOT)/research/fuzz,$(PY) fixed_rule_loss.py)

# ---- full: long accuracy runs ----------------------------------------------
referee-r:
	$(call run,referee-r,$(CURDIR),$(PY) referee_cases.py && $(RSCRIPT) referee.R && $(PY) referee_check.py)
botev-tree:   # tree tails below MC resolution vs minimax tilting (needs R)
	$(call run,botev-tree,$(ROOT)/research/tree_exactness,$(RSCRIPT) botev_tree.R)
ensembles:
	$(call run,ensembles,$(ROOT)/research/general_sigma,$(PY) run_ensembles4.py)
kernel:
	$(call run,kernel,$(ROOT)/research/general_sigma,$(PY) run_kernel4.py)
estimation:
	$(call run,estimation,$(ROOT)/research/mnp_estimation,$(PY) run_mle2.py --reps 40 --out $(OUT)/results_40.json)
covariates:
	$(call run,covariates,$(ROOT)/research/mnp_estimation,$(PY) run_mle3.py && $(PY) run_mle4.py)
independent-inversion:
	$(call run,independent-inversion,$(ROOT)/research/experiments/exp35_independent_inversion,$(PY) run_independent.py)
large-n:
	$(call run,large-n,$(ROOT)/research/general_sigma,$(PY) run_large_n.py)
newton-krylov:
	$(call run,newton-krylov,$(ROOT)/research/experiments/exp23_newton_cg,$(PY) run_newton_cg3.py)
blp-ablation:
	$(call run,blp-ablation,$(ROOT)/research/inversion_ablation,$(PY) run_blp_ablation.py)
factor-rqmc:
	$(call run,factor-rqmc,$(ROOT)/research/experiments/exp24_factor_rqmc,$(PY) run_factor_rqmc.py)
law:
	$(call run,law,$(CURDIR),$(PY) bench.py law)
scale:
	$(call run,scale,$(CURDIR),$(PY) bench.py scale)
tenmillion:   # about 5 GB of memory
	$(call run,tenmillion,$(CURDIR),$(PY) bench.py tenmillion)
invertmillion:
	$(call run,invertmillion,$(CURDIR),$(PY) bench.py invertmillion)

# ---- wall clock -------------------------------------------------------------
timings:
	@mkdir -p $(OUT)
	cd $(CURDIR) && $(PY) retime.py --json $(OUT)/retime.json | tee $(OUT)/timings.log

clean:
	rm -rf $(OUT)
