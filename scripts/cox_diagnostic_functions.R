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

#########################################################
##Extract loans that are active as of curr_date
extractCurrentLoans <- function(curr_date,
                                dt,
                                time_to_status,
                                status){
    
    curr_date = as.numeric(curr_date)
    
    approval_date = as.Date(dt$ApprovalDate,origin='1970-01-01')

    loan_paid_date = as.POSIXlt(approval_date)#this is just to add years directly
    loan_paid_date$year = as.POSIXlt(approval_date)$year + 20
    loan_paid_date = as.numeric(as.Date(loan_paid_date))    

    approval_date = as.numeric(approval_date)
    
    charge_off_date = ifelse(status, approval_date + time_to_status, NA)
    
    cond1 = approval_date < curr_date
    cond2 = loan_paid_date > curr_date
    cond3 = ifelse(!is.na(charge_off_date), charge_off_date > curr_date, TRUE)

    print(paste('loans not yet approved: ', sum(!cond1), sep=''))
    print(paste('loans paid off before current date: ', sum(!cond2), sep=''))
    print(paste('loans defaulted before current date: ', sum(!cond3), sep=''))
    print(paste('number of active loans as of ',
                as.Date(curr_date,origin='1970-01-01'),
                ': ', sum(cond1&cond2&cond3), sep = ''))

    ##Extract these from the data
    dt = dt[cond1 & cond2 & cond3,]
    dt
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

##Calculates probability of default for data.frame of new loans
##between individual loan age and individual loan age + time_ahead.
pOfDefaultOverNext.multObs <- function(fit_mod, new_data, loan_ages, time_ahead){    

    require(parallel)
    ## Calculate the number of cores
    no_cores <- detectCores() - 1    
    ## Initiate cluster
    cl <- makeCluster(no_cores,
                      outfile = paste('probability_calcs_',
                                      round(time_ahead,2),'.txt',sep=''))

    warningMessage <- function(loan_age,time_ahead){
        paste('NOTE: This loan is ', round(loan_age/365.25,2),
              ' years old, and you are trying to predict',
              ' default probability over the next ',
              round(time_ahead/365.2) ,
              ' years. All loans have term of 20 years. Returning NA.',
              sep='')
    }
    
    iterateMessage <- function(idx,loan_age,time_ahead){
        print(paste(idx,': getting prob of default over next ',
                    time_ahead, ' days for loan of age ',
                    round(loan_age/365.25,2) ,
                    ' years', sep = ''))    
    }
    
    innerWorkings <- function(idx, fit_mod, new_data, loan_ages, time_ahead){
        iterateMessage(idx,loan_ages[idx],time_ahead)
        if((loan_ages[idx] + time_ahead)/365.25 > 20 ){
            warningMessage(loan_ages[idx],time_ahead)
            return(NA)
        }
        
        else{
            return(pOfDefaultBtwn(fit_mod,
                                  new_data[idx,],
                                  loan_ages[idx],
                                  loan_ages[idx] + time_ahead))
        }
    }
    
    clusterExport(cl, varlist = c("warningMessage","iterateMessage",
                                  "innerWorkings","fit_mod","new_data",
                                  "loan_ages","time_ahead","pOfDefaultBtwn",
                                  "dt_train","train_surv_obj"),
                  envir = environment())
    
    clusterEvalQ(cl, library(peperr))
    clusterEvalQ(cl, library(survival))
    out = parLapply(cl, 1:nrow(new_data), innerWorkings,
                    fit_mod = fit_mod,
                    new_data = new_data,
                    loan_ages = loan_ages,
                    time_ahead = time_ahead)

    stopCluster(cl)
    return(out)
}




##Predict probability of default between t1 and t2 (loan age)
##This gives general S(t1) - S(t2) = P(t1 < T < t2) 
#pOfDefaultBtwn(cox_fit, dt_test_curr[1:100,], 1000, 7000)


