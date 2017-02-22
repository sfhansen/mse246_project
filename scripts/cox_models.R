##########################################################
##Fit Penalized Cox Models For Various lambda and alpha
require(data.table)
require(glmnet)
require(survival)

load('../data/cox_data_environment_train.RData')

alpha_seq = seq(0,1,by=0.1)
lambda_seq = seq(0.2,0.000001,by=-0.00001)

fitCoxModels <- function(time_to_status,status,
                         dt,alpha_seq,lambda_seq){
    out_list = list()
    for(alpha in alpha_seq){
        print(paste('fitting model for alpha ',alpha,sep=''))
        mod_name = paste('alpha_',alpha,sep='')
        mod = glmnet(x = dt,
                     y = Surv(time_to_status, status
                              ),
                     lambda = lambda_seq,
                     alpha = alpha,
                     family = 'cox'
                     )
        out_list[[mod_name]] = mod
    }
    return(out_list)
}

print('fitting 20yr models...')
fitted_mods = fitCoxModels(
    time_to_status,
    status,
    dt,
    alpha_seq,
    lambda_seq)

rm(list = setdiff(ls(),c('alpha_seq',
                         'lambda_seq',
                         'fitted_mods')))

save.image(file='../data/cox_models_glmnet_fitted.RData')

