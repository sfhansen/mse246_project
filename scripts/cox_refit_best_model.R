require(data.table)
require(glmnet)
require(survival)

load('../data/cox_models_glmnet_fitted.RData') #list of glmnet models
load('../data/cox_data_environment_train.RData') #load training data
source('./cox_diagnostic_functions.R')

####################################################
##Select best model from penalized models. Refit with coxph function.

##Selects best penalized model:
best_mod = selectBestCox(fitted_mods)

##Select variables from data corresponding to those selected by penalized model:
dt_train = data.frame(selectNonZeroVars(best_mod,dt))

##This is a surv object for use in coxph:
train_surv_obj = Surv(time_to_status,
                      status)

##This is the refit model:
cox_fit = coxph(train_surv_obj ~ .,
                data = dt_train)


rm(list = setdiff(ls(),c('best_mod',
                         'train_surv_obj',
                         'dt_train',
                         'cox_fit')))

save.image(file = '../data/cox_refit_best_model.RData')

