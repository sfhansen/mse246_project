require(data.table)
require(ggplot2)
require(glmnet)
require(survival)

load('../data/cox_models_glmnet_fitted.RData')
dt_10yr_test = data.table(dt_10yr_test)
dt_20yr_test = data.table(dt_20yr_test)
dt_10yr_train = data.table(dt_10yr_train)
dt_20yr_train = data.table(dt_20yr_train)

##########################################################
##Functions to plot dev.ratio by alpha and lambda

heatMapDevRatio <- function(glmnet_obj_list,lambda_seq,alpha_seq){
    grid <- matrix(NA,nr=1,ncol=3)    
    colnames(grid) = c('alpha','lambda','dev.ratio')
    for(idx in 1:length(glmnet_obj_list)){
        alpha = getAlpha(names(glmnet_obj_list)[idx])
        temp_df = data.frame(alpha=alpha,
                             lambda=lambda_seq,
                             dev.ratio=glmnet_obj_list[[idx]]$dev.ratio)
        grid = rbind(grid,temp_df)
    }
    grid = grid[-1,]
    ggplot(grid, aes(lambda, alpha)) +
        geom_raster(aes(fill = dev.ratio), interpolate = TRUE)
}

##########################################################
##Functions to extract info from a list of glmnet model outputs

##Extracts alpha value from model name, that's where I put that info...
getAlpha <- function(name){
    as.numeric(regmatches(name,
                          regexpr('(?<=alpha_)[0-9]*\\.[0-9]*',
                                  name,
                                  perl=T)))
}

##Find index of model with largest dev.ratio in glmnet_obj_list
getMaxDevRatioModelIdx <- function(glmnet_obj_list){
    max_dr_vec = numeric(0)
    for(obj in glmnet_obj_list){
        local_max_dr_idx = which(obj$dev.ratio==max(obj$dev.ratio))
        local_max_dr = obj$dev.ratio[local_max_dr_idx]
        max_dr_vec = c(max_dr_vec,local_max_dr)
    }
    max_dr_obj_idx = which(max_dr_vec==max(max_dr_vec)) ##model idx in glmnet_obj_list
    return(max_dr_obj_idx)
}

##Use dev.ratio from glmnet to select best model from model list 
selectBestCox <- function(glmnet_obj_list){
    max_idx = getMaxDevRatioModelIdx(glmnet_obj_list)
    obj = glmnet_obj_list[[max_idx]]

    dr_idx = which(obj$dev.ratio==max(obj$dev.ratio))
    
    model_name = names(glmnet_obj_list)[max_idx]
    coefficients = obj$beta[,dr_idx] ##coeff. corresp. to max dev.ratio
    no_vars = sum(coefficients>0)
    lambda = obj$lambda[dr_idx]
    alpha = getAlpha(model_name)
    dev_ratio = obj$dev.ratio[dr_idx]
    
    print('Best model by max dev ratio:')
    print(paste('alpha: ',alpha,sep=''))
    print(paste('lambda: ',lambda,sep=''))
    print(paste('dev.ratio:',dev_ratio,sep=''))
    print(paste('number of vars selected: ',no_vars,sep=''))

    return(glmnet_obj_list[[max_idx]])
}

##########################################################
##Fits a new model with Survival package based on best model
##from glmnet output. This is important in order to estimate
##survival curve.

fitSurvivalCox <- function(best_glm_mod, time_to_default,
                           default, dt_train){
    
    ##get column names of vars that are nonzero in glmnet mod
    l_idx = which(max(best_glm_mod$dev.ratio) == best_glm_mod$dev.ratio)
    coefficients = best_glm_mod$beta[,l_idx]
    non_zero_idx = which(coefficients>0)
    var_names = rownames(best_glm_mod$beta)[non_zero_idx]

    ##select data columns that correspond to non-zero coeffs
    dt_train_new = dt_train[,var_names,with=F]

    
    new_mod = coxph(Surv(time_to_default,default)~.,
                    data=dt_train_new,
                    iter=0,
                    init=coefficients[non_zero_idx])
    
    return(new_mod)
}

##########################################################
##Applying these functions

print('heatmaps...')
png('../studies/cox_models_heatmap_10yr.png')
heatMapDevRatio(fitted_mods_10yr[2:10],lambda_seq,alpha_seq)
dev.off()

png('../studies/cox_models_heatmap_20yr.png')
heatMapDevRatio(fitted_mods_20yr[2:10],lambda_seq,alpha_seq)
dev.off()

print('select best...')
best_mod_10yr = selectBestCox(fitted_mods_10yr)
best_mod_20yr = selectBestCox(fitted_mods_20yr)

cox_fit_10yr = fitSurvivalCox(best_mod_10yr,
                              time_to_default_10yr[train_10yr_idx],
                              default_10yr[train_10yr_idx],
                              dt_10yr_train)

cox_fit_20yr = fitSurvivalCox(best_mod_20yr,
                              time_to_default_20yr[train_20yr_idx],
                              default_20yr[train_20yr_idx],
                              dt_20yr_train)

surv_curve_10yr = survfit(cox_fit_10yr)
surv_curve_20yr = survfit(cox_fit_20yr)

rm(list = setdiff(ls(),c('cox_fit_10yr',
                         'cox_fit_20yr',
                         'surv_curve_10yr',
                         'surv_curve_20yr')))

save.image(file='../data/cox_models_survival_curves.RData')
