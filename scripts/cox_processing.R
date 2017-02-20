##Takes command line arguments:

##For training data out run:
##Rscript cox_processing.R ../data/train_with_exempt.rds ../data/cox_data_environment_train.RData

##For test data out run:
##Rscript cox_processing.R ../data/test_with_exempt.rds ../data/cox_data_environment_test.RData
require(data.table)

args = commandArgs(TRUE)

dt_in = args[1] #rds file with either training or test data
env_out = args[2] #RData file with environment, appropriately named

dt = data.table(readRDS(dt_in))

##########################################################
##Creation of Cox Variables
##Loan ages should all be in days...

##Loan term
term = dt[,TermInMonths]
termInYears = as.numeric(term)/12 #years
chgOffDate = dt$ChargeOffDate #date
apprvlDate = dt$ApprovalDate #date
censorDate = as.Date('2/01/2014','%m/%d/%Y') #date

loanPaidOffDate = as.POSIXlt(apprvlDate) #this is just to add years directly
loanPaidOffDate$year = as.POSIXlt(apprvlDate)$year + termInYears
loanPaidOffDate = as.Date(loanPaidOffDate)

##status: 0-right censored, 1-paid off, 2-default
status_m = ifelse(!is.na(chgOffDate), 2, #else...
           ifelse(loanPaidOffDate <= censorDate, 1, #else...
                0))

time_to_status = ifelse(!is.na(chgOffDate), chgOffDate - apprvlDate,
                 ifelse(loanPaidOffDate <= censorDate, loanPaidOffDate - apprvlDate,
                        censorDate - apprvlDate))

##Recode the statuses as 0-right-censored/paid-off, 1-default
status = ifelse((status_m == 0) | (status_m == 1), 0, 1)

##########################################################
##Exclude loans with term other than 20yrs

##This is categorical, it's to separate the dataset later
loanTermCat = ifelse(term=="240","20yr",ifelse(term=="120","10yr","exclude"))

dt = copy(dt[loanTermCat=='20yr',])
status = status[loanTermCat=='20yr']
time_to_status = time_to_status[loanTermCat=='20yr']

##########################################################
##These variables are outcomes.
remove = c('ChargeOffDate','LoanStatus','GrossChargeOffAmount')
dt[, (remove) := NULL]

##########################################################
##Additional Processing Functions

standardizeUnits <- function(dt){
    col_names = colnames(dt)
    ncol_dt = ncol(dt)
    for(idx in 1:ncol_dt){
        col <- dt[, get(col_names[idx])]
        if(class(col) == 'numeric'){
            dt[, (col_names[idx]) := (col-mean(col,na.rm=T))/sd(col,na.rm=T)]
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
##Only save these items:
rm(list = setdiff(ls(),c('env_out',
                         'dt',
                         'status',                         
                         'time_to_status',
                         'status_m')))
save.image(file = env_out)
