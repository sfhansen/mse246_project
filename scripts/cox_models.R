require(data.table)
require(glmnet)
require(survival)

merged_data_in = '../data/merged.csv'
dt = fread(merged_data_in)
print(head(dt))
print(dim(dt))
print(sapply(dt,class))

##########################################################
##Preliminary Processing

##Coerce 'N/A's to proper NAs, recognize as dates
##(as.numeric turns them into days since 1/1/1970)
dt[, ChargeOffDate := as.numeric(as.Date(ChargeOffDate, "%m/%d/%y"))]
dt[, ApprovalDate := as.numeric(as.Date(ApprovalDate, "%Y-%m-%d"))]

##wrong NA form
dt[SameThirdPartyLendingState == "N/A", SameThirdPartyLendingState := NA]

##just make this a 0-1 indicator, "504" is meaningless as an integer
dt[, DeliveryMethod := as.integer(ifelse(!is.na(DeliveryMethod),1,NA))]

##want these to be numeric so that polynom() will see them
bump_to_num = c('ThirdPartyDollars','gdp',
                'GrossApproval','TermInMonths',
                'GrossChargeOffAmount')
dt[, (bump_to_num) := lapply(.SD, as.numeric), .SDcols = bump_to_num]

##make these character so that they will be correctly interp. as categorical
bump_to_char = c('NAICS')
dt[, (bump_to_char) := lapply(.SD, as.character), .SDcols = bump_to_char]

##Default indicator to use in the cox regression
default = dt[, ifelse(is.na(ChargeOffDate),0,1)]
##days, life of the loan
time_to_default = dt[, ChargeOffDate - ApprovalDate]
##The 'follow up time' is to take the place of entries that did not default in
##time_to_default. So in this case, that would be the end of the window.
S = as.numeric(as.Date('2/01/2014','%m/%d/%Y')) #right censor date, window end + 1 day
censor_obs_idx = which(is.na(time_to_default))
time_to_default[censor_obs_idx] = S - dt[censor_obs_idx, ApprovalDate]

##I don't think these variables are useful (anymore, or ever)
remove = c('ChargeOffDate','BorrZip','FiscalYear','ProjectState','BorrState')
dt[, (remove) := NULL] 

##Check again that types are correct:
print('----------numeric----------')
print(names(which(sapply(dt,class)=='numeric')))
print('----------character----------')
print(names(which(sapply(dt,class)=='character')))
print('----------integer----------')
print(names(which(sapply(dt,class)=='integer')))
print('----------logical----------')
print(names(which(sapply(dt,class)=='logical')))
##########################################################
##Additional Processing Functions

##Makes units of all numeric cols in (-1,1],
##Divides them by max value
standardizeUnits <- function(dt){
    col_names = colnames(dt)
    ncol_dt = ncol(dt)
    for(idx in 1:ncol_dt){
        col <- dt[, get(col_names[idx])]
        if(class(col) == 'numeric'){
            dt[, (col_names[idx]) := col/max(col,na.rm=T)]
        }
    }
}

##Determines if a vector is 0-1 indicator
isIndicator <- function(col){
    all(unique(col) %in% c(1,0,NA))   
}

##Replaces NA values with 0, creates a 0-1 dummy variable in dt 
dummifyNAs <- function(dt){
    col_names = colnames(dt)
    for(col_name in col_names){
        if(dt[,sum(is.na(get(col_name)))]!=0){
            na_dum_name <- paste('na_dum_',
                                 col_name,
                                 sep='')
            dt[, (na_dum_name) := ifelse(is.na(get(col_name)),1,0)]
            dt[is.na(get(col_name)), (col_name) := 0]
        }
    }
}

##Adds n-1 polynomial features for numeric columns
addPolynomialFeatures <- function(dt, n){
    col_names = colnames(dt)
    ncol_dt <- ncol(dt)
    for(idx in 1:ncol_dt){
        col <- dt[, get(col_names[idx])]
        if(class(col) == 'numeric' & !isIndicator(col)){
            for(poly_idx in 2:n){
                poly_col_name = paste('p_',
                                      poly_idx,
                                      '_',
                                      col_names[idx],
                                      sep='')
                dt[, (poly_col_name) := col^(poly_idx)]
            }
        }
    }
}

standardizeUnits(dt)
##Are all 'numeric' cols in (-1,1]?
sapply(dt[, which(sapply(dt,class)=='numeric'), with=F],summary)
  
dummifyNAs(dt)
##Are there any NAs left? 

addPolynomialFeatures(dt,5)
##Were columns added for the right variables?

##########################################################
##Models

##glmnet doesn't seem to want to automagically convert char to factor to indicator.
##So I did that here.
dt = model.matrix(~.,data=dt)

alpha_seq = seq(0.5,1,by=0.1)
lambda_seq = seq(0.2,0.0001,by=-0.0005)

fitCoxModels <- function(dt,alpha_seq,lambda_seq){
    out_list = list()
    for(alpha in alpha_seq){
        print(paste('fitting model for alpha ',alpha,sep=''))
        mod_name = paste('alpha_',alpha,'_mod',sep='')
        ##cv.glmnet uses partial likelihood to det. best fit for cox
        ##10 folds default
        mod = cv.glmnet(x = dt,
                        y = Surv(time_to_default, default),
                        lambda = lambda_seq,
                        alpha = alpha,
                        family = 'cox',
                        maxit = 10000)
        out_list[[mod_name]] = mod
    }
    return(out_list)    
}

fitted_mods = fitCoxModels(dt,alpha_seq,lambda_seq)
