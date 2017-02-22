####################################################
##Predict survival probabilities on test data.
load('../data/cox_refit_best_model.RData')
load('../data/cox_data_environment_test.RData')
source('cox_diagnostic_functions.R')

dt_test = data.frame(selectNonZeroVars(best_mod,dt))
dt_test = cbind(ID=1:nrow(dt_test),dt_test)

##The date of the portfolio construction
curr_date = as.Date('2014-02-01','%Y-%m-%d')

##Extract currently active loans from test set.
dt_test_curr = extractCurrentLoans(curr_date,
                                   dt_test,
                                   time_to_status,
                                   status)

saveRDS(dt_test_curr, file='../data/test_current_loans_2014.RDS')

##Half year
horizon = 365.25/2 
loan_age = as.numeric(curr_date) - dt_test_curr$ApprovalDate
p_0.5 = pOfDefaultOverNext.multObs(cox_fit,
                                   dt_test_curr,
                                   loan_age,
                                   horizon)

##One year
horizon = 365.25
loan_age = as.numeric(curr_date) - dt_test_curr$ApprovalDate
p_1 = pOfDefaultOverNext.multObs(cox_fit,
                                 dt_test_curr,
                                 loan_age,
                                 horizon)

##Five years
horizon = 365.25*5
loan_age = as.numeric(curr_date) - dt_test_curr$ApprovalDate
p_5 = pOfDefaultOverNext.multObs(cox_fit,
                                 dt_test_curr,
                                 loan_age,
                                 horizon)

save.image(file = '../data/cox_survival_probabilities.RData')
