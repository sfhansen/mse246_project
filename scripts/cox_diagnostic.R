require(data.table)
require(ggplot2)
require(glmnet)
require(survival)
require(peperr)

load('../data/cox_models_glmnet_fitted.RData') #list of glmnet models
load('../data/cox_data_environment.RData') #load data sets
source('./cox_diagnostic_functions.R')

##########################################################
##Applications:

##Selects best penalized model:
best_mod = selectBestCox(fitted_mods_20yr)

##Select variables from data corresponding to those selected by penalized model:
dt_train = data.frame(selectNonZeroVars(best_mod,dt_train))
dt_test = data.frame(selectNonZeroVars(best_mod,dt_test))

##This is a surv object for use in coxph:
train_surv_obj = Surv(time_to_status[train_idx],
                      status[train_idx])

##This is the refit model:
cox_fit = coxph(train_surv_obj ~ .,
                data = dt_train)

##Predict probability of default between 1000 days and 7000 days (general loan age).
##This gives general S(t1) - S(t2) = P(t1 < T < t2) 
pOfDefaultBtwn(cox_fit, dt_test[1:100,], 1000, 7000)

##Probability of default over next 200 days, given current age.
##You need to give a vector of current loan ages as third arg.
##If the loan is too old (near 7300 days), will return NA.
##This takes a long time to run ~3min for 100 obs...
##I will think about how to make it faster...
pOfDefaultOverNext.multObs(cox_fit, #fit coxph model
                           dt_test[1:100,], #new data
                           time_to_status_test[1:100], #loan age
                           200)

