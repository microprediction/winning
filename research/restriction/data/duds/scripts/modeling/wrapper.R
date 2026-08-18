options(dplyr.summarise.inform = FALSE)
options(tidyverse.quiet = TRUE)
library(tidyverse, warn.conflicts = FALSE)
library(DirichletReg, warn.conflicts = FALSE)
source('simTrials.R')
source('fit.R')
source('computeLogLikelihoodDecisions.R')
source('computeLogLikelihoodConfidence.R')
source('loopOnConditionsD.R')
source('loopOnConditionsC.R')
source('compareModelvsData.R')


# this script fits the computational models from the paper "The presence 
# of irrelevant alternatives paradoxically increases confidence in perceptual
# decisions" by Comay, Della Bella, Lamberti, Sigman, Solovey & Barttfeld.


####----  1) load data & choose model  ----####
data <- read.csv2('data_experiment1.csv')

# 'av_res' (Average-residual model); 'contrast' (Contrast model);
# 'max' (Max model); 'diff' (Diff model).
model <- 'av_res'


####----  2) fixed parameters  ----####
# sizes of the alternatives
a1 <- 1
a2 <- c(0.7,0.8,0.9,0.93,0.95)   # levels of difficulty (a2 size)
a3 <- 0.245                      # a3 mean size

# number of simulations
nd <- 10000  # n of simulations for fitting decisions
n  <- 1000   # n of simulations for fitting confidence
m  <- 1000   # n of means of confidence


####----  3) fit  ----####
fittedpars <- fit(data,model,a1,a2,a3,nd,n,m) 


####----  4) compare model predictions with data (plot) ----####
compareModelvsData(data,fittedpars,model,n,m,a1,a2,a3)

