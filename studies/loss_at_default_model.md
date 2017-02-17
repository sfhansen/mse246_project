# Loss at Default Model
Samuel Hansen  
1/21/2017  




```r
# Initialize libraries 
library(knitr)
library(caret)
library(stringr)
library(tidyverse)

# Initialize input files 
train_file_in = "../data/train.rds"
portfolio_file_in = "../data/portfolio.rds"
rf_file_in = "../models/rf.fit.rds"

# Read in data 
train = read_rds(train_file_in)
portfolio = read_rds(portfolio_file_in)
```

#Data Preparation

##Cleaning

Before fitting the loss at default model, we clean the training set 
by filtering it to only include defaulted loans and by removing unnecessary
features such as '`LoanStatus`. 


```r
# Prepare train df to only include defaulted loans 
train_with_defaults = 
  train %>%
  # Filter to only include defaulted loans
  filter(LoanStatus == "default") %>%
  # Remove unecessary variables 
  select(-c(LoanStatus, ChargeOffDate, ApprovalDate, first_zip_digit))
```

We extract the `GrossChargeOffAmount` mean and standard deviation for 
later re-normalization. 


```r
# Extract GrossChargeOffAmount mean and standard deviation
mean_GrossChargeOffAmount = mean(train_with_defaults$GrossChargeOffAmount)
sd_GrossChargeOffAmount = sd(train_with_defaults$GrossChargeOffAmount)
```

##Pre-processing

We pre-process the data by centering and scaling all continuous features. 
We apply the same normalization from the training data with only defaulted loans
to the portfolio data used for prediction in the loss at default model. 
Similarly, we apply the same training normalization from the entire training 
set to the portfolio data used for prediction in the default probability model.


```r
# Apply pre-processing steps to the data
preProcessSteps = c("center", "scale", "nzv")

# Apply pre-processing steps from training set with only defaults
preProcessObject_LAD = preProcess(train_with_defaults, method = preProcessSteps)
train_with_defaults = predict(preProcessObject_LAD, train_with_defaults)

# Apply same pre-processing to portfolio for loss at default model 
portfolio_LAD = predict(preProcessObject_LAD, portfolio)

# Apply pre-processing steps from training set with all data 
preProcessObject_probs = preProcess(train, method = preProcessSteps)
train = predict(preProcessObject_probs, train)

# Apply same pre-processing to portfolio for default probability model 
portfolio_probs = predict(preProcessObject_probs, portfolio)
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
  rfe(GrossChargeOffAmount~.,
      data = train_with_defaults,
      rfeControl = rfe.cntrl,
      preProc = preProcessSteps,
      sizes =  seq(12,132,10),
      metric = "ROC",
      trControl = train.cntrl)

# Map factor levels back to their respective features 
selected_vars <- map(predictors(rfe.results), 
                      ~str_match(.x, names(train_with_defaults))) %>% 
  unlist() %>% 
  .[!is.na(.)] %>%
  unique())
train_selected_vars <- train %>%
  select(one_of(selected_vars), GrossChargeOffAmount)
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

##Elastic Net

We fit an elastic net model as follows:


```r
# # Define grid of tuning parameters
elasticGrid = expand.grid(.alpha = seq(0, 1, 0.1),
                         .lambda = seq(0, 0.05, by = 0.005))

# Fit penalized linear regression model (elastic net)
set.seed(1234)
elastic.fit = train(GrossChargeOffAmount ~ .,
                   data = train_selected_vars,
                   preProc = preProcessSteps,
                   method = "glmnet",
                   # tuneGrid = elasticGrid, 
                   trControl = cvCtrl)
```

#Model Evaluation 

##Test Set Prediction  

We calculate the expected loss and default probability of each loan in the 
portfolio of 500 loans by using the model of expected loss and best model
of default probability. 


```r
# Read in default probability model 
rf.fit = read_rds(rf_file_in)

# Make data frame with expected loss and default probability for each loan
prediction_df = 
  data_frame(default_loss = predict(elastic.fit, portfolio_LAD),
             default_prob = predict(rf.fit, portfolio_probs, 
                                    type = "prob")[, "default"]) 
 
  ## Renormalize loss amounts into dollars 
  # mutate(default_loss = sd_GrossChargeOffAmount * default_loss +
  #          mean_GrossChargeOffAmount) 
  ## Ensure loss is zero or above 
  # mutate(default_loss = ifelse(default_loss < 0, 0, default_loss))
```

##Simulate Distribution of Total Loss 

```r
# Initialize simulation parameters 
num_simulations = 1000
num_loans = nrow(prediction_df)
result <- vector("numeric", num_simulations)

# run through simulations for the 500 loans
for (simulation in c(1:num_simulations)) {
  loss = 0  
  
  # generate a uniform random variable and comparing that to our prediction 
  uniform_probs <- runif(nrow(num_loans))
  for (loan in c(1:nrow(num_loans))) {
    if (uniform_probs[loan] < prediction_df$default_prob[loan]) {
      
      # if the uniform rv is smaller than our prediction, it has defaulted
      loss = loss + prediction_df$default_loss[loan]
    }  
  }
  
  # each entry in the result vector is the simulated loss for the 500 samples combined
  result[simulation] = loss
}
```



```r
# plotting the histogram for the losses 
result = data.frame(result)
ggplot(data = result, aes(result)) +
  geom_histogram(position = "identity", bins = 100) +
  # scale_x_continuous(labels = scales::dollar) +
  labs(x = "Total Loss at Default from 500 Loans", y = "Count", 
       title = "Distribution of Total Loss from 500 Loan Portfolio")
```

