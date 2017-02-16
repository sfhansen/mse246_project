require(data.table)
require(ggplot2)
require(glmnet)
require(survival)
require(peperr)

##########################################################
##Function to plot dev.ratio by alpha and lambda
heatMapDevRatio <- function(glmnet_obj_list,
                            alpha_seq,
                            plot_name){

    grid <- matrix(NA,nr=1,ncol=3)    
    colnames(grid) = c('alpha','lambda','dev.ratio')
    for(idx in 1:length(glmnet_obj_list)){
        alpha = getAlpha(names(glmnet_obj_list)[idx])
        temp_df = data.frame(alpha=alpha,
                             lambda=glmnet_obj_list[[idx]]$lambda,
                             dev.ratio=glmnet_obj_list[[idx]]$dev.ratio)
        grid = rbind(grid,temp_df)
    }
    grid = grid[-1,]

    ggplot(data = grid, aes(lambda, alpha)) +
        geom_raster(aes(fill = dev.ratio), interpolate = TRUE) +
        ggtitle(plot_name) +
        xlab(expression(lambda)) +
        ylab(expression(alpha))
}

##########################################################
##Functions to extract info from a list of glmnet model outputs.

##Extracts alpha value from model name, that's where I put that info...
getAlpha <- function(name){
    as.numeric(regmatches(name,
                          regexpr('(?<=alpha_)[0-9]*\\.*[0-9]*',
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

selectNonZeroVars <- function(best_glm_mod,dt){
    ##get column names of vars that are nonzero in glmnet mod
    l_idx = which(max(best_glm_mod$dev.ratio) == best_glm_mod$dev.ratio)
    coefficients = best_glm_mod$beta[,l_idx]
    non_zero_idx = which(coefficients>0)

    ##select data columns that correspond to non-zero coeffs
    dt_new = dt[,non_zero_idx]

    return(dt_new)
}

##########################################################
##Default Probability Functions

##Gives probability of default for data.frame of new loans, 'new_data' between
##loan ages of t1 and t2. 
pOfDefaultBtwn <- function(fit_mod, new_data, t1, t2){
    n = nrow(new_data)
    p.mat = predictProb(fit_mod,
                        Surv(rep(1,n),rep(0,n)),
                        new_data, c(t1,t2))
    p.btwn.t1.t2 = p.mat[,1] - p.mat[,2]

    p.btwn.t1.t2   
}

##For a single *active* loan that has not defaulted, with age 'loan_age', this
#calculates probability that the loan will default over next time_ahead (days) amount of time 
pOfDefaultOverNext.singleObs <- function(fit_mod, active_loan, loan_age, time_ahead){
    pOfDefaultBtwn(fit_mod, active_loan, loan_age, loan_age + time_ahead)
}

##This takes forever... hope to make more efficient. It's the application of the previous function to multiple observations.
pOfDefaultOverNext.multObs <- function(fit_mod, new_data, loan_ages, time_ahead){    
    p_of_def = matrix(NA,nr=nrow(new_data),nc=1)
    for(idx in 1:nrow(new_data)){
        p_of_def[idx] = pOfDefaultOverNext.singleObs(fit_mod,
                                                     new_data[idx,],
                                                     loan_ages[idx],
                                                     loan_ages[idx] + time_ahead)
    }
    p_of_def
}
