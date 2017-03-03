#!/bin/bash

##Set wd
cd /home/theodorevadpey/Documents/Classes/MSE246/project/mse246_project/scripts/
#########################################################
##TRAINING PROCESSING
echo "processing training data..."
Rscript cox_processing.R ../data/train_with_exempt.rds ../data/cox_data_environment_train.RData
echo "created file cox_data_environment_train.RData with training data ready for modeling..."
#########################################################
##TEST PROCESSING
echo "processing test data..."
Rscript cox_processing.R ../data/test_with_exempt.rds ../data/cox_data_environment_test.RData
echo "created file ../data/cox_data_environment_test.RData with test data ready for modeling"
#########################################################
##SELECT PORTFOLIO
echo "selecting a portfolio of 500 valid loans from test set as of 2010-02-01..."
Rscript cox_processing.R -p ../data/test_with_exempt.rds ../data/portfolio.rds
echo "created file ../data/portfolio.rds containing raw portfolio loan data"
#########################################################
##PORTFOLIO TEST PROCESSING
echo "processing portfolio test data..."
Rscript cox_processing.R ../data/portfolio.rds ../data/portfolio_test_environment.RData
echo "created file ../data/portfolio_test_environment.RData containing portfolio test data ready for modeling"
#########################################################
##FIT PENALIZED COX MODELS
echo "fitting penalized cox models on training data, varying alpha and lambda..."
Rscript cox_models.R
#########################################################
##SELECT BEST MODEL
echo "select best model from fit models in last step, refitting..."
Rscript cox_refit_best_model.R
#########################################################
##CALCULATE 1, 5 YEAR PROBABILITY OF DEFAULT FOR PORTFOLIO
echo "calculating 1, 5 year probabilities of default for portfolio test data + make ROC graphs..."
Rscript cox_surv_probs.R ../data/portfolio_test_environment.RData ../data/cox_portfolio_probabilities.RData
#########################################################
##CALCULATE 1, 5 YEAR PROBABILITY OF DEFAULT FOR ALL TEST
echo "calculating 1, 5 year probabilities of default for ALL test data + make ROC graphs..."
Rscript cox_surv_probs.R -t ../data/cox_data_environment_test.RData ../data/cox_test_probabilities.RData
#########################################################
##REMAKE ROC GRAPHS, DON'T RECALCULATE PROBABILITIES
echo "remake ROC graphs for test data..."
Rscript cox_surv_probs.R -t -g ../data/cox_data_environment_test.RData ../data/cox_test_probabilities.RData
#########################################################

Rscript -e "rmarkdown::render('cox_summary.Rmd', output_format = 'pdf_document', output_file = 'cox_summary[exported].pdf')"
