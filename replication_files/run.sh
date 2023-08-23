#!/usr/bin/env bash
set -ex

# This is the master script for the capsule.

python3 1-iv_bounds.py

python3 2-outcome_selection.py

python3 3-measurement_error.py

python3 4-nonresponse.py
scip -c "set limits gap 0.01 read results/nonresponse.pip optimize quit" >results/nonresponse_max.log
scip -c "set limits gap 0.01 read results/nonresponse.pip change objsense min optimize quit" >results/nonresponse_min.log

python3 5-nonresponse2.py
scip -c "set limits gap 0.04 read results/nonresponse2.pip optimize quit" >results/nonresponse2_max.log
scip -c "set limits gap 0.04 read results/nonresponse2.pip change objsense min optimize quit" >results/nonresponse2_min.log


python3 6-mediation.py

python3 7-iv-more-categories.py

python3 8-ci.py

# Running plots
Rscript 9-plots.R

Rscript 10-ci-visualization.R
