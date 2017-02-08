##########################################################
##Fit Models
require(data.table)
require(glmnet)
require(survival)

load('../data/cox_data_environment.RData')
print(ls())

alpha_seq = seq(0,1,by=0.1)
lambda_seq = seq(0.5,0.00001,by=-0.0002)

fitCoxModels <- function(time_to_default,default,
                         dt,alpha_seq,lambda_seq){
    out_list = list()
    for(alpha in alpha_seq){
        print(paste('fitting model for alpha ',alpha,sep=''))
        mod_name = paste('alpha_',alpha,sep='')
        mod = glmnet(x = dt,
                     y = Surv(time_to_default, default),
                     lambda = lambda_seq,
                     alpha = alpha,
                     family = 'cox')
        out_list[[mod_name]] = mod
    }
    return(out_list)
}

print('fitting 10yr models...')
##note to index default, time_to_default by train_idx
fitted_mods_10yr = fitCoxModels(
    time_to_default_10yr[train_10yr_idx],
    default_10yr[train_10yr_idx],
    dt_10yr_train,
    alpha_seq,
    lambda_seq)

print('fitting 20yr models...')
fitted_mods_20yr = fitCoxModels(
    time_to_default_20yr[train_20yr_idx],
    default_20yr[train_20yr_idx],
    dt_20yr_train,
    alpha_seq,
    lambda_seq)

save.image(file='../data/cox_models_glmnet_fitted.RDat')

