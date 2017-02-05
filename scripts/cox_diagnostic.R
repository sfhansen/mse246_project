require(data.table)
require(ggplot2)
require(glmnet)

load('../data/cox_models.RDat')
dt_test_in = '../data/cox_models_data.test.csv'

alpha_seq = seq(0,1,by=0.1)
lambda_seq = seq(0.5,0.00001,by=-0.0002)

dt_test = fread(dt_test_in)
##########################################################
##Select Best Model, Diagnostics

diagnostic <- function(glmnet_obj_list){
    idx = 1
    for(obj in glmnet_obj_list){
        print(names(glmnet_obj_list)[idx])
        print(length(obj$lambda))
        print(length(obj$dev.ratio))
        print(dim(obj$beta))
        idx = idx + 1
    }
}

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

#Makes line plots of lambda vs dev.ratio for each alpha
plotLambdaByDevRatio <- function(glmnet_obj_list,plot_dim){
    list_names = names(glmnet_obj_list)
    par(mfrow = plot_dim)
    idx = 1
    for(obj in glmnet_obj_list){
        with(obj,
             plot(lambda,dev.ratio,type='l',lwd=3,
                  main=list_names[idx],ylim=c(0,.35))
             )
        idx = idx + 1
    }
}

##Makes heatmap of lambda vs. alpha vs. dev.ratio
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

diagnostic(fitted_mods)

print('heatmaps...')
png('../studies/cox_models_heatmap.png')
heatMapDevRatio(fitted_mods[2:10],lambda_seq,alpha_seq)
dev.off()

print('plots...')
png('../studies/cox_models_linegraphs.png')
plotLambdaByDevRatio(fitted_mods[2:11],c(2,5))
dev.off()

print('select best...')
best_mod = selectBestCox(fitted_mods)

##########################################################
##Make Predictions

out = predict(best_mod,newx=as.matrix(dt_test[1:10,]))

dim(out) #10 x 2500
