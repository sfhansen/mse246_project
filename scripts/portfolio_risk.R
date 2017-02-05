require(data.table)
require(glmnet)
require(survival)
require(ggplot2)

merged_data_in = '~/Dropbox/0.Stanford/0.Courses/0.Quarter-5/MS&E 246/mse246_project/data/merged.csv'
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
bump_to_char = c('NAICS','FiscalYear')
dt[, (bump_to_char) := lapply(.SD, as.character), .SDcols = bump_to_char]

##Default indicator to use in the cox regression
default = dt[, ifelse(is.na(ChargeOffDate),0,1)]
default_idx <- which(default == 1)

##days, life of the loan
time_to_default = dt[, ChargeOffDate - ApprovalDate]
View(as.data.frame(dt))
##The 'follow up time' is to take the place of entries that did not default in
##time_to_default. So in this case, that would be the end of the window.
S = as.numeric(as.Date('2/01/2014','%m/%d/%Y')) #right censor date, window end + 1 day
censor_obs_idx = which(is.na(time_to_default))
time_to_default[censor_obs_idx] = S - dt[censor_obs_idx, ApprovalDate]


#need to normalize chargeoff??
severity <- dt[, GrossApproval] - dt[, GrossChargeOffAmount]
mean_chargeoff <- mean(dt[which(dt[, GrossChargeOffAmount]>0), GrossChargeOffAmount])
mean_loan_size <- mean(dt[which(dt[, GrossApproval]>0), GrossApproval])

#head(which(severity <0 ))

##I don't think these variables are useful (anymore, or ever)
remove = c('ChargeOffDate','BorrZip','ProjectState')
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


#putting in dataframe format
dt2 = as.data.frame(dt)

print('done with preprocessing...')
##########################################################
length(time_to_default)
length(default)
y <- Surv(time_to_default, default)
summary(y)

#random simulation of right censored data
# lifetimes <- rexp( 25, rate = 0.2)
# censtimes <- 5 + 5*runif(25)
# ztimes <- pmin(lifetimes, censtimes)
# status <- as.numeric(censtimes > lifetimes)

#Non parametric estimation of distirbution of default
y1 <- survfit(Surv(time_to_default, default)~1) 
plot(y1, xlab = "Time", ylab = "Surv proba")

#Parametric estimation of distirbution of default
y3 <- survreg(Surv(time_to_default, default)~as.data.frame(dt))

pred <- predict(y3, type="quantile", p=c(0.1, 0.5, 0.9) )


#TODO: 
#estimate default probability --> theta
#generate bernoulli simulation with different theta params
#generate 


