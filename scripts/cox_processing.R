##This script takes the data merged.csv as input.
##As output, it returns an R environment containing all variables required to feed into a cox model.
##--It also splits the data into 10yr and 20yr loan-term training and test sets-- 4 data sets in all.
##--It aslo standarzes numeric variables, creates NA dummy features, and creates polynomial features.

require(data.table)

dt = fread('../data/merged.csv')

##########################################################
##Preliminary Processing

##Coerce 'N/A's to proper NAs, recognize as dates
##(as.numeric turns them into days since 1/1/1970)
dt[, ChargeOffDate := as.Date(ChargeOffDate, "%m/%d/%y")]
dt[, ApprovalDate := as.Date(ApprovalDate, "%Y-%m-%d")]

##wrong NA form
dt[SameThirdPartyLendingState == "N/A", SameThirdPartyLendingState := NA]

##just make this a 0-1 indicator, "504" is meaningless as an integer
dt[, DeliveryMethod := as.integer(ifelse(!is.na(DeliveryMethod),1,NA))]

##want these to be numeric so that polynom() will see them
bump_to_num = c('ThirdPartyDollars','gdp',
                'GrossApproval','TermInMonths')
dt[, (bump_to_num) := lapply(.SD, as.numeric), .SDcols = bump_to_num]

##make these character so that they will be correctly interp. as categorical
bump_to_char = c('NAICS','FiscalYear','TermInMonths')
dt[, (bump_to_char) := lapply(.SD, as.character), .SDcols = bump_to_char]

##########################################################
##Creation of Cox Variables
##Loan ages should all be in days...

##Loan term
term = dt[,TermInMonths]
termInYears = as.numeric(term)/12 #years
chgOffDate = dt[,ChargeOffDate] #date
apprvlDate = dt[,ApprovalDate] #date
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

##IMPORTANT:
##For now, treat paid off as right-censored.
##If you wish to keep them coded differently, delete this line.
status = ifelse((status_m == 0) | (status_m == 1), 0, 1)

##This is categorical, it's to separate the dataset later
loanTermCat = ifelse(term=="240","20yr",ifelse(term=="120","10yr","exclude"))

##########################################################
##I don't think these variables are useful (anymore, or ever)
remove = c('ChargeOffDate','BorrZip','ProjectState','TermByYear','TermInMonths',
           'LoanStatus','GrossChargeOffAmount')
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

print('done with initial stuff...')
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
##Split Data by Loan Term
##It is important to split the loans into groups with similar terms for
##the hazard modeling. Otherwise we would be modeling survival of objects with
##deterministically dissimilar lifespans.

dt = copy(dt[loanTermCat=='20yr',])
status = status[loanTermCat=='20yr']
time_to_status = time_to_status[loanTermCat=='20yr']

##########################################################
##Training Test Split

set.seed(10)
train_idx = sample(1:nrow(dt),floor(0.7*nrow(dt)),replace=F)

dt_train = dt[train_idx,]
dt_test = dt[-train_idx,]

status_train = status[train_idx]
status_test = status[-train_idx]
time_to_status_train = time_to_status[train_idx]
time_to_status_test = time_to_status[-train_idx]

##Only save these items:
rm(list = setdiff(ls(),c('dt_train',
                         'dt_test',
                         'status_train',                         
                         'time_to_status_train',
                         'status_test',
                         'time_to_status_test',
                         'status_m')))

save.image(file='../data/cox_data_environment.RData')
