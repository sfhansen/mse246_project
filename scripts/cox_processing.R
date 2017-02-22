##Takes command line arguments:

##Process training data:
##Rscript cox_processing.R ../data/train_with_exempt.rds ../data/cox_data_environment_train.RData

##Process test data:
##Rscript cox_processing.R ../data/test_with_exempt.rds ../data/cox_data_environment_test.RData

##Create portfolio subset of the test data
##Rscript cox_processing.R -p ../data/test_with_exempt.rds ../data/portfolio.rds

require(optparse)
source('cox_diagnostic_functions.R')

option_list = list(
    make_option(c("-p", "--portfolio_option"), action="store_true", default=FALSE,
                help="output portfolio of 500 active loans from test input"))

parser = OptionParser(usage="%prog [options] file", option_list=option_list)

args = parse_args(parser, positional_arguments = 2)
opt = args$options
in_out = args$args

dt_in = in_out[1] #rds file with either training or test data
env_out = in_out[2] #RData file with environment, appropriately named

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
##Select portfolio of 500 here if option is flagged
if(opt$portfolio_option){
    dt = data.frame(dt)

    set.seed(353)
    sample_idx_1 = sample(which(status==1),250,replace=F)
    sample_idx_0 = sample(which(status==0),500,replace=F)
    sample_idx = union(sample_idx_1, sample_idx_0)
    
    dt = dt[sample_idx,]
    status = status[sample_idx]
    time_to_status = time_to_status[sample_idx]
    
    dt = extractCurrentLoans(as.Date('2010-02-01','%Y-%m-%d'),
                             dt,
                             time_to_status,
                             status)
    saveRDS(dt,file=env_out)
}else{

    ##All other processing steps:
    ##########################################################    
    ##These variables are outcomes. Should be removed.
    remove = c('ChargeOffDate','LoanStatus','GrossChargeOffAmount')
    dt[, (remove) := NULL]    
    
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
}
