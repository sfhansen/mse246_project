##Parse options, arguments
require(optparse)

option_list = list(
    make_option(c("-t", "--using_test_data"), action="store_true", default=FALSE,
                help="operate on full test data rather than portfolio test data"),
    make_option(c("-g", "--only_make_graphs"), action="store_true", default=FALSE,
                  help="probabilities already saved just make graphs"))

parser = OptionParser(usage="%prog [options] file", option_list=option_list)

args = parse_args(parser, positional_arguments = 2)
opt = args$options
args = args$args

test_data_in = args[1]
out_file = args[2]

####################################################
##Predict survival probabilities on test data.
load('../data/cox_refit_best_model.RData')
load(test_data_in) 
source('cox_diagnostic_functions.R')
require(caTools)

dt_test_curr = data.frame(selectNonZeroVars(best_mod,dt))

##Portfolio formation date
curr_date = as.Date('2010-02-01','%Y-%m-%d')

loan_age = as.numeric(curr_date) - dt_test_curr$ApprovalDate

if(opt$using_test_data){
    ##The portfolio test data already met all these criteria.
    ##So if you want to use the general test data, you have to filter
    ##the loans in the same way as in cox_processing.R for the portfolio.
    dt_test_curr$ApprovalDate = as.Date(dt_test_curr$ApprovalDate,
                                        origin='1970-01-01')
    
    ##Need to keep track of idx of which loans get extracted
    dt_test_curr = cbind(IDX = 1:nrow(dt_test_curr), dt_test_curr)
    
    dt_test_curr = extractCurrentLoans(curr_date,
                                       dt_test_curr,
                                       time_to_status,
                                       status)
    
    ##Make sure all of these match with the extracted loans.
    status = status[dt_test_curr$IDX]
    time_to_status = time_to_status[dt_test_curr$IDX]
    loan_age = loan_age[dt_test_curr$IDX]

    ##Get only loans with age less than 15 years.
    young_idx = which(loan_age < 365.25*15)
    dt_test_curr = dt_test_curr[young_idx,]
    status = status[young_idx]
    time_to_status = time_to_status[young_idx]
    loan_age = loan_age[young_idx]
    
    ##Make dt_test_curr like it was before this if block
    dt_test_curr = dt_test_curr[,-1]
    dt_test_curr$ApprovalDate = as.numeric(dt_test_curr$ApprovalDate)
}

if(!opt$only_make_graphs){
    ##One year
    horizon = 365.25
    p_1 = pOfDefaultOverNext.multObs(cox_fit,
                                     dt_test_curr,
                                     loan_age,
                                     horizon)

    ##Five years
    horizon = 365.25*5
    p_5 = pOfDefaultOverNext.multObs(cox_fit,
                                     dt_test_curr,
                                     loan_age,
                                     horizon)

    save.image(file = '../data/intermediate_prob_out.RData')    
}else{
    get_stuff_from_env <- function(obj_name,rdata_file){
        load('../data/intermediate_prob_out.RData')
        return(eval(parse(text=obj_name)))
    }
    p_1 = get_stuff_from_env('p_1','../data/intermediate_prob_out.RData')
    p_5 = get_stuff_from_env('p_5','../data/intermediate_prob_out.RData')
}

##ROC curves, implemented all myself, in cox_diagnostic_functions.R
status_1yr = ifelse(((time_to_status - loan_age) < 365.25) & status, 1, 0)
status_5yr = ifelse(((time_to_status - loan_age) < 365.25*5) & status, 1, 0)

thresh = seq(0,0.61,by=0.001)

p_1_predict_mtx = populate_predict_mtx(p_1,thresh)
p_5_predict_mtx = populate_predict_mtx(p_5,thresh)

fpr_1 = calculate_fpr(p_1_predict_mtx, status_1yr) 
tpr_1 = calculate_tpr(p_1_predict_mtx, status_1yr) 

fpr_5 = calculate_fpr(p_5_predict_mtx, status_5yr) 
tpr_5 = calculate_tpr(p_5_predict_mtx, status_5yr) 


x_ord = order(fpr_1)
auc_1 = trapz(fpr_1[x_ord],tpr_1[x_ord])

png('../studies/p_1_roc_curve.png')
plot(fpr_1,tpr_1,type='l',cex = 0.85,lwd = 3,
     ylim = c(0,1),xlim = c(0,1),
     xlab = 'FPR', ylab = 'TPR',
     main = 'ROC curve for 1 year ahead default predictions')
polygon(c(sort(fpr_1),max(fpr_1),0),c(sort(tpr_1),0,0),col='grey')
abline(a = 0, b = max(tpr_1)/max(fpr_1),
       col='gray60',lty=3,lwd=2)
mtext(paste('AUC:',round(auc_1,3)*100,'%',sep=''),side=1,line=-2,cex=1.5)
dev.off()


x_ord = order(fpr_5)
auc_5 = trapz(fpr_5[x_ord],tpr_5[x_ord])

png('../studies/p_5_roc_curve.png')
plot(fpr_5,tpr_5,type='l',cex = 0.85,lwd = 3,
     ylim = c(0,1), xlim = c(0,1),
     xlab = 'FPR', ylab = 'TPR',
     main = 'ROC curve for 5 year ahead default predictions')
polygon(c(sort(fpr_5),max(fpr_5),0),c(sort(tpr_5),0,0),col='grey')
abline(a = 0, b = max(tpr_5)/max(fpr_5),
       col='gray60',lty=3,lwd=2)
mtext(paste('AUC:',round(auc_5,3)*100,'%',sep=''),side=1,line=-2,cex=1.5)
dev.off()


rm(list = setdiff(ls(),c('p_1',
                         'p_5',
                         'auc_1',
                         'auc_5',
                         'out_file')))

save.image(file = out_file)
