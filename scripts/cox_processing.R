require(data.table)

merged_data_in = '../data/merged.csv'
dt = fread(merged_data_in)

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
bump_to_char = c('NAICS','FiscalYear','TermInMonths')
dt[, (bump_to_char) := lapply(.SD, as.character), .SDcols = bump_to_char]

##Create default, time_to_default for use in Cox model

##Default indicator to use in the cox regression
default = dt[, ifelse(is.na(ChargeOffDate),0,1)]

##days, life of the loan
time_to_default = dt[, ChargeOffDate - ApprovalDate]

##The 'follow up time' is to take the place of entries that did not default in
##time_to_default. So in this case, that would be the end of the window.
##right censor date, S, window end + 1 day
S = as.numeric(as.Date('2/01/2014','%m/%d/%Y'))
censor_obs_idx = which(is.na(time_to_default))
time_to_default[censor_obs_idx] = S - dt[censor_obs_idx, ApprovalDate]

##This will be useful later-- after preprocessing I will split the data by loan term.
term = dt[,TermInMonths]
loanTermCat = ifelse(term=="240","20yr",ifelse(term=="120","10yr","exclude"))

##I don't think these variables are useful (anymore, or ever)
remove = c('ChargeOffDate','BorrZip','ProjectState','TermByYear','TermInMonths')
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

print('done with preprocessing...')
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

print('standardizing units...')
standardizeUnits(dt)

print('adding na dummies... will have warnings, its ok...')
dummifyNAs(dt)

print('adding polynomial ftrs...')
addPolynomialFeatures(dt,n=5)

print('dummifying character variables...')
dt = model.matrix(~.,data=dt)

##########################################################
##Split Data by Loan Term
##It is important to split the loans into groups with similar terms for
##the hazard modeling. Otherwise we would be modeling survival of objects with
##deterministically dissimilar lifespans.

dt_20yr = copy(dt[loanTermCat=='20yr',])
dt_10yr = copy(dt[loanTermCat=='10yr',])

default_20yr = default[loanTermCat=='20yr']
default_10yr = default[loanTermCat=='20yr']

time_to_default_20yr = time_to_default[loanTermCat=='20yr']
time_to_default_10yr = time_to_default[loanTermCat=='10yr']

##########################################################
##Training Test Split

set.seed(100)
train_10yr_idx = sample(1:nrow(dt_10yr),floor(0.7*nrow(dt_10yr)),replace=F)

set.seed(10)
train_20yr_idx = sample(1:nrow(dt_20yr),floor(0.7*nrow(dt_20yr)),replace=F)

dt_10yr_train = dt_10yr[train_10yr_idx,]
dt_10yr_test = dt_10yr[-train_10yr_idx,]

dt_20yr_train = dt_20yr[train_20yr_idx,]
dt_20yr_test = dt_20yr[-train_20yr_idx,]

##Only save these items:
rm(list = setdiff(ls(),c('dt_10yr_train',
                         'dt_10yr_test',
                         'dt_20yr_train',
                         'dt_20yr_test',
                         'train_10yr_idx',
                         'train_20yr_idx',
                         'default_10yr',
                         'default_20yr',                         
                         'time_to_default_10yr',
                         'time_to_default_20yr')))

save.image(file='../data/cox_data_environment.RData')
