require(caret)
require(data.table)
require(glmnet)
require(survival)

merged_data_in = '../data/merged.csv'

dt = fread(merged_data_in)

print(head(dt))
print(dim(dt))

##Coerce 'N/A's to proper NAs, recognize as dates
dt[, ChargeOffDate := as.Date(ChargeOffDate, "%m/%d/%y")]
dt[, ApprovalDate := as.Date(ApprovalDate, "%Y-%m-%d")]

default = dt[,ifelse(is.na(ChargeOffDate),0,1)]
timeToDefault = dt[,ChargeOffDate - ApprovalDate] #days

addPolynomialFeatures <- function(dt, n){
    col_names = colnames(dt)
    col_classes = sapply(dt, class)
    for(col_idx in 1:ncol(dt)){
        if(col_classes[col_idx] == 'numeric'){
            for(poly_idx in 1:n){
                poly_col_name = paste('p_',
                                      poly_idx,
                                      '_',
                                      col_names[col_idx],
                                      sep='')
                dt[,(poly_col_name) := get(col_names[col_idx])^(poly_idx)]
            }
        }
    }
}

standardizeUnits <- function(dt){
    col_names = colnames(dt)
    col_classes = sapply(dt, class)
    for(col_idx in 1:ncol(dt)){
        if(col_classes[col_idx] == 'numeric'){
            dt[,(col_names[col_idx]) :=
                    get(col_names[col_idx])/max(get(col_names[col_idx]))]
        }
    }    
}

dummifyNAs <- function(dt){
    col_names = colnames(dt)
    for(col_name in col_names){
        if(dt[,sum(is.na(get(col_name)))]!=0){
            na_dum_name <- paste('na_dum_',
                                 col_name,
                                 sep='')
            dt[,(na_dum_name) := ifelse(is.na(get(col_name)),1,0)]
            dt[is.na(get(col_name)), (col_name) := 0]
        }
    }
}

standardizeUnits(dt)
dummifyNAs(dt)
addPolynomialFeatures(dt, 5)

##ok, now use caret to find the right alpha, right lambda
##pick the best model from various penalized regressions 

mod = glmnet(x = dt_data,
             y = Surv(timeToDefault,
                      default),
             nlambda = 1000,
             alpha = alpha,
             family = 'cox',
             maxit = 10000)

