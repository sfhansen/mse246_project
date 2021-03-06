# Loss at Default Model
Samuel Hansen  
2/23/2017  




```r
# Initialize libraries 
library(knitr)
library(caret)
library(PerformanceAnalytics)
library(stringr)
library(tidyverse)

# Initialize input files 
train_file_in = "../data/train.rds"
portfolio_file_in = "../data/portfolio.rds"
prob_file_in <- "../data/cox_portfolio_probabilities.RData"

rf.fit_file = '../data/alex_rf_fit.rds'
prediction_df_file = '../data/alex_prediction_df.rds'
etl_file = '../data/alex_etl_list.rds'
var_file = '../data/alex_var_list.rds'


# Read in data 
train = read_rds(train_file_in)
portfolio = read_rds(portfolio_file_in)
load(prob_file_in)
```

#Data Preparation

##Cleaning

Before fitting the loss at default model, we clean the training set 
by filtering it to only include defaulted loans and by removing unnecessary
features such as `LoanStatus`. 


```r
# Prepare train data frame to only include defaulted loans 
cols_to_drop_train  = c("LoanStatus", "ChargeOffDate", "ApprovalDate", 
                  "first_zip_digit", "GrossChargeOffAmount", "GrossApproval")

train_with_defaults = 
  train %>%
  # Filter to only include defaulted loans
  filter(LoanStatus == "default") %>%
  # Make percentage loss column 
  mutate(percent_loss = GrossChargeOffAmount/GrossApproval) %>%
  # Filter out percent loss above 1
  filter(percent_loss <= 1) %>%
  # Remove unecessary variables 
  select(-one_of(cols_to_drop_train))

cols_to_drop_df  = c("LoanStatus", "ChargeOffDate", "ApprovalDate", 
                  "first_zip_digit", "GrossChargeOffAmount")
# Prepare portfolio data frame to match training variables 
portfolio = 
  portfolio %>%
  # Make percentage loss column 
  mutate(percent_loss = GrossChargeOffAmount/GrossApproval) %>%
  # Remove unecessary variables 
  select(-one_of(cols_to_drop_df)) 

# Convert probabilities object to data frame 
default_probs = 
  data_frame(one_year_prob = p_1 %>% unlist(),
             five_year_prob = p_5 %>% unlist()) 
```

##Pre-processing

We pre-process the data by centering and scaling all continuous features. 
We apply the same normalization from the training data with only defaulted loans
to the portfolio data used for prediction in the loss at default model. 
Similarly, we apply the same training normalization from the entire training 
set to the portfolio data used for prediction in the default probability model.


```r
# Define pre-processing steps
preProcessSteps = c("nzv")
```

#Feature Selection

To select the features that are used in the loss at default model, 
we perform recursive feature elimination. 


```r
# Set the recursive feature elimination parameters 
set.seed(1234)
rfe.cntrl = rfeControl(functions = rfFuncs,
                      method = "cv",
                      number = 5)
train.cntrl = trainControl(selectionFunction = "oneSE")

# Perform recursive feature elimination to select variables
rfe.results =
  rfe(percent_loss~.,
      data = train_with_defaults,
      rfeControl = rfe.cntrl,
      preProc = preProcessSteps,
      sizes =  seq(12,132,10), # commented out to reduce runtime 
      metric = "ROC",
      trControl = train.cntrl)

# Map factor levels back to their respective features 
selected_vars <- map(predictors(rfe.results), 
                      ~str_match(.x, names(train_with_defaults))) %>% 
  unlist() %>% 
  .[!is.na(.)] %>%
  unique()

# Create data frame with these selected variables 
train_selected_vars <- train_with_defaults %>%
  select(one_of(selected_vars), percent_loss)
```

#Model Fitting 

To tune hyperparameters, we use 5-fold cross-validation with the "one standard 
error" rule. 


```r
# Define cross-validation controls 
cvCtrl = trainControl(method = "cv", 
                       number = 5,
                       selectionFunction = "oneSE",
                       classProbs = TRUE)
                       # allowParallel = TRUE)
```

##Random Forest Model 

```r
# Define grid of tuning parameters
rfGrid = expand.grid(.mtry = c(2, 4, 6, 8, 10, 14))
# Fit random forest model
set.seed(1234)
rf.fit = train(percent_loss ~ .,
                   data = train_selected_vars,
                   preProc = preProcessSteps,
                   method = "rf",
                   tuneGrid = rfGrid, # commented out to reduce runtime
                   trControl = cvCtrl)

rf.fit_file = '../data/alex_rf_fit.rds'
write_rds(rf.fit, rf.fit_file)
```

#Model Evaluation 

##Test Set Prediction  

We calculate the expected loss and default probability of each loan in the 
portfolio of 500 loans by using the model of expected loss and best model
of default probability. 


```r
# Make data frame with predicted loss and default probability for each loan
prediction_df = 
  bind_cols(
    # Predicted losses from loss at default model 
    data_frame(default_loss = predict(rf.fit, portfolio)) %>%
      # Apply sigmoid transformation to recover values between 0-1 
      mutate(default_loss = 1 / (1 + exp(1)^(-default_loss))),
    # Bind 1- and 5-year default probabilities
    default_probs) 

#Add GrossApproval for later computation of total loss
prediction_df$GrossApproval <-  portfolio$GrossApproval

prediction_df_file = '../data/alex_prediction_df.rds'
write_rds(prediction_df, prediction_df_file)
 
# USE THIS CODE IF LOSS AT DEFAUL MODEL PREDICTS DOLLAR AMOUNTS
  # ## Renormalize loss amounts into dollars 
  # mutate(default_loss = sd_GrossChargeOffAmount * default_loss +
  #          mean_GrossChargeOffAmount) %>%
  # ## Ensure loss is zero or above 
  # mutate(default_loss = ifelse(default_loss < 0, 0, default_loss))
```

##Simulate Distribution of Total Loss 
To estimate the value at risk, we generate simulations of the loan losses 
for the portfolio in batches. For each batch of 10000 portfolio simulations, 
we compute the value at risk and expected shortfall and store them. We then
take the average value at risk and calculate confidence interval for both 
metrics.


```r
# Initialize simulation parameters 
num_simulations = 10000
num_sim_batch = 100
num_loans = nrow(prediction_df)
var1yr_vec95 <- vector("numeric", num_sim_batch)
var1yr_vec99 <- vector("numeric", num_sim_batch)
var5yr_vec95 <- vector("numeric", num_sim_batch)
var5yr_vec99 <- vector("numeric", num_sim_batch)
etl1yr_vec95 <- vector("numeric", num_sim_batch)
etl1yr_vec99 <- vector("numeric", num_sim_batch)
etl5yr_vec95 <- vector("numeric", num_sim_batch)
etl5yr_vec99 <- vector("numeric", num_sim_batch)

for (j in c(1:num_sim_batch)){
  
  one_year_default_result <- vector("numeric", num_simulations)
  five_year_default_result <- vector("numeric", num_simulations)
  
  # run through simulations for the 500 loans
  for (simulation in c(1:num_simulations)) {
    one_year_loss = 0  
    five_year_loss = 0  
    
    # generate a uniform random variable and comparing that to our prediction 
    uniform_probs <- runif(num_loans)
    for (loan in c(1:num_loans)) {
      
       # if the uniform rv is smaller than one year default prob, loan has defaulted
      if (uniform_probs[loan] < prediction_df$one_year_prob[loan]) {
        one_year_loss = one_year_loss + prediction_df$default_loss[loan] * prediction_df$GrossApproval[loan]
      } 
      
       # if the uniform rv is smaller than five year default prob, loan has defaulted
      if (uniform_probs[loan] < prediction_df$five_year_prob[loan]) {
        five_year_loss = five_year_loss + prediction_df$default_loss[loan] * prediction_df$GrossApproval[loan]
      } 
    }
    
    # Update result vectors with total loss of current simulation 
    one_year_default_result[simulation] = one_year_loss
    five_year_default_result[simulation] = five_year_loss
  }
  
  # Convert result objects to data frames 
  portfolio_nominal <- sum(prediction_df$GrossApproval)
  one_year_default_result = data.frame(DollarAmount = one_year_default_result, 
                                       Percentage = one_year_default_result/portfolio_nominal)
  five_year_default_result = data.frame(DollarAmount = five_year_default_result, 
                                        Percentage = five_year_default_result/portfolio_nominal)
  
  #Compute Value at risk for every simulation batch
  var1yr_vec95[j] <- -VaR(-one_year_default_result$Percentage, p=0.95)[1]
  var1yr_vec99[j] <- -VaR(-one_year_default_result$Percentage, p=0.99)[1]
  var5yr_vec95[j] <- -VaR(-five_year_default_result$Percentage, p=0.95)[1]
  var5yr_vec99[j] <- -VaR(-five_year_default_result$Percentage, p=0.99)[1]
  
  #Compute expected shortfall for each batch
  etl1yr_vec95[j] <- -ETL(-one_year_default_result$Percentage, p=0.95)[1]
  etl1yr_vec99[j] <- -ETL(-one_year_default_result$Percentage, p=0.99)[1]
  etl5yr_vec95[j] <- -ETL(-five_year_default_result$Percentage, p=0.95)[1]
  etl5yr_vec99[j] <- -ETL(-five_year_default_result$Percentage, p=0.99)[1]
  cat("\nSim: ",j, " done")
}
```

##Plot Expected Loss Distributions

```r
# Plot the histogram for the 1-year losses 
ggplot(data = one_year_default_result, aes(one_year_default_result$Percentage)) +
  geom_histogram(position = "identity") +
  labs(x = "Total Loss at Default from 500 Loans", y = "Count", 
       title = "One-Year Expected Loss Distribution from 500 Loan Portfolio")
```




```r
# Plot the histogram for the 5-year losses 
ggplot(data = five_year_default_result, aes(five_year_default_result)) +
  geom_histogram(position = "identity") +
  labs(x = "Total Loss at Default from 500 Loans", y = "Count", 
       title = "Five-Year Expected Loss Distribution from 500 Loan Portfolio")
```

##Compute Value-at-Risk
We measure the risk in terms of the 1 year and 5 years ahead VaR at the 95% and 99% levels and include confidence intervals.


```r
#Data frame containing VaR metrics for each simulation
var1yr = data.frame(
  var1yr_vec95,
  var1yr_vec95*portfolio_nominal,
  var1yr_vec99,
  var1yr_vec99*portfolio_nominal)

var5yr = data.frame(
  var5yr_vec95,
  var5yr_vec95*portfolio_nominal,
  var5yr_vec99,
  var5yr_vec99*portfolio_nominal)

#list containing global VaR metrics
var1yr_95 <- list("Mean"= mean(var1yr_vec95),"Sd" = sd(var1yr_vec95))
var1yr_95$CI <- c(var1yr_95$Mean - 1.96 * var1yr_95$Sd / num_sim_batch,
                  var1yr_95$Mean + 1.96 * var1yr_95$Sd / num_sim_batch)

var1yr_99 <- list("Mean"= mean(var1yr_vec95),"Sd" = sd(var1yr_vec99))
var1yr_99$CI <- c(var1yr_99$Mean - 1.96 * var1yr_99$Sd / num_sim_batch,
                  var1yr_99$Mean + 1.96 * var1yr_99$Sd / num_sim_batch)

var5yr_95 <- list("Mean"= mean(var5yr_vec95),"Sd" = sd(var5yr_vec95))
var5yr_95$CI <- c(var5yr_95$Mean - 1.96 * var5yr_95$Sd / num_sim_batch,
                  var5yr_95$Mean + 1.96 * var5yr_95$Sd / num_sim_batch)

var5yr_99 <- list("Mean"= mean(var5yr_vec99),"Sd" = sd(var5yr_vec99))
var5yr_99$CI <- c(var5yr_99$Mean - 1.96 * var5yr_99$Sd / num_sim_batch,
                  var5yr_99$Mean + 1.96 * var5yr_99$Sd / num_sim_batch)

VaR_list <- list(var1yr, var5yr, var1yr_95, var1yr_99, var5yr_95, var5yr_99)
var_file = '../data/alex_var_list.rds'
write_rds(VaR_list, var_file)
```

##Compute Average Value-at-Risk

We also measure the risk in terms of the 1 year and 5 years ahead verage VaR at the 95% and 99% levels and include confidence intervals


```r
ETL1yr = data.frame(
  etl1yr_vec95,
  etl1yr_vec95*portfolio_nominal,
  etl1yr_vec99,
  etl1yr_vec99*portfolio_nominal)

ETL5yr = data.frame(
  etl5yr_vec95,
  etl5yr_vec95*portfolio_nominal,
  etl5yr_vec99,
  etl5yr_vec99*portfolio_nominal)

etl1yr_95 <- list("Mean"= mean(etl1yr_vec95),"Sd" = sd(etl1yr_vec95))
etl1yr_95$CI <- c(etl1yr_95$Mean - 1.96 * etl1yr_95$Sd / num_sim_batch,
                  etl1yr_95$Mean + 1.96 * etl1yr_95$Sd / num_sim_batch)

etl1yr_99 <- list("Mean"= mean(etl1yr_vec95),"Sd" = sd(etl1yr_vec99))
etl1yr_99$CI <- c(etl1yr_99$Mean - 1.96 * etl1yr_99$Sd / num_sim_batch,
                  etl1yr_99$Mean + 1.96 * etl1yr_99$Sd / num_sim_batch)

etl5yr_95 <- list("Mean"= mean(etl5yr_vec95),"Sd" = sd(etl5yr_vec95))
etl5yr_95$CI <- c(etl5yr_95$Mean - 1.96 * etl5yr_95$Sd / num_sim_batch,
                  etl5yr_95$Mean + 1.96 * etl5yr_95$Sd / num_sim_batch)

etl5yr_99 <- list("Mean"= mean(etl5yr_vec99),"Sd" = sd(etl5yr_vec99))
etl5yr_99$CI <- c(etl5yr_99$Mean - 1.96 * etl5yr_99$Sd / num_sim_batch,
                  etl5yr_99$Mean + 1.96 * etl5yr_99$Sd / num_sim_batch)
ETL_list <- list(ETL1yr, ETL5yr, etl1yr_95, etl1yr_99, etl5yr_95, etl5yr_99)
etl_file = '../data/alex_etl_list.rds'
write_rds(ETL_list, etl_file)
```
