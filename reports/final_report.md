# MS&E 246 Final Report
Samuel Hansen, Theo Vadpey, Alex Elkrief, Ben Ertringer  
2/23/2017  





#Exectutive Summary

In *MS&E 246: Financial Risk Analytics*, our team analyzed a data set of 
roughly 150,000 loans backed by the US Small Business Administration 
(SBA) between 1990 and 2014. In doing so, we aimed to implement and test models
of the risk and loss of loan default. This report summarizes our findings from exploratory data analysis, details our approaches to modeling loan 
default probability and loss, and presents our methods of estimating
the loss distributions of tranches backed by a 500-loan portfolio. 

#Exploratory Data Analysis

Prior to model building, we explored the data to detect patterns that may 
provide signal for models of loan default. Because we first aimed to build 
binary response models of default probability, we excluded "Exempt" loans from 
our exploratory analysis. Subsequently, we examined the relationship between 
default rates and the predictor variables, including `Business Type`, 
`Loan Amount`, `NAICS Code`, and `Loan Term`, among others. 

Further, we collected additional predictor variables such as monthly 
`GDP`, `Crime Rate`, and `Unemployment Rate` by State, as well as macroeconomic
predictors such as monthly measures of the `S&P 500`, `Consumer Price Index`, 
and 14 other volatility market indices (see Data Cleaning section for 
data collection details). We include insights from exploratory analysis of 
these measures as well. 

##Default Rate vs. Business Type 

First, we examined the relationship between default rate and `Business Type`
by loan approval year. As shown on the plot below, we observe an interaction
effect between these three features, such that default rates spiked for 
loans that were approved around the Great Recession (approximately 2006- 2009). 
Further, the different trajectories of the 3 curves implies the "individual" 
`Business Type` suffered greater default rates than corporations and 
partnerships. Although corporations constitute a greater share of the data set,
as evidenced by the greater mass in the red circles, they exhibit medium 
default risk, as compared to the other business types. Taken together, 
this plot reveals business types were affected differently by the recession,
offering useful signal for subsequent modeling. 

![](final_report_files/figure-html/unnamed-chunk-2-1.png)<!-- -->

##Default Rate by Loan Amount

Second, we examined whether we would observe a similar time-dependent 
interaction effect between default rate and `Loan Amount`. The plot below 
reveals that loans of all sizes approved around the Great Recession faced the
greatest default rates. However, loans of sizes \$500k-\$1m and \$1m-\$2m
appear to have experienced larger default rates over time compared to smaller
loans of size \$100k-\$300k and \$300k-\$500k. The spiking behavior of \$1m-\$2m
loan in 1999 and of loans greater than \$2m seem to be due to small sample 
sizes, as depicted by circle diameter. Overall, since loans of different sizes
have different default rate patterns over time, we would also expect 
the `Loan Amount` feature to offer predictive power. 

![](final_report_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

##Default Rate by NAICS Code

Third, we hypothesized different economic sectors would exhibit 
different default rates over time. In turn, we extracted the North American 
Industry Classification System (NAICS) code for each loan and truncated it
to the first two digits, which represents broad industry classes such as 
"Agriculture" and "Manufacturing." The following plot shows the default 
rate for loans of each truncated NAICS code approved in each year between 
1990-2014. We observe considerable variance in default rates between sectors;
for instance, codes 72, corresponding to "Accommodation & Food Services", 
has one of the highest default rates even before the recession. However,
code 54, corresponding to "Professional, Scientific, and Technical Services,"
consistently has one the lowest default rates. These patterns are consistent
with intuition, and underscore the value of including the truncated NAICS code
as a predictive feature of defaulting. 

![](final_report_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

##State GDP vs. Default Rate

- Make plot here

#Modeling Default Probability 

Building upon our exploratory data analysis, we constructed two types of 
predictive models of loan default probability: binary response models and 
cox proportional hazards models. Here, we present our approach to fitting
both model types, including data cleaning, feature engineering, 
feature selection, hyper-parameter optimization, and evaluation. 

##Binary Response Models

First, we built binary response models of small-businesses defaulting on loans,
which estimate the probability that a given loan *ever* defaults. To do so,
we implemented a machine learning pipeline that: 

1. Performs feature engineering;
2. Splits the data into train and test sets;
3. Normalizes continuous features;
4. Selects features using recursive feature elimination;
5. Trains binary response predictive models. 

Lastly, we evaluate the performance of these models on resampled partitions 
of the training data, and on a held-out test set in terms of AUC, sensitivity, 
and calibration. 

###Feature Engineering

Building on insights derived from exploratory data analysis,
we engineered the following features from the raw data: 

- `NAICS_code`: truncated to the first two digits of the NAICS code;
- `subprogram`: condensed infrequent factor levels into "other" category;
- `approval_year`: extracted year from loan approval date-time object.
- `SameLendingState`: created flag for whether borrower received loan from in-state; 
- `MultiTimeBorrower`: created flag for whether loan recipient is multi-time borrower;
- `ThirdPartyLender` created flag for whether borrower received third party aide. 

In effect, these features represent dimensionality reduction of factors 
with many levels. For instance, there are 1,239 unique NAICS six-digit NAICS
codes in the raw data, yet only 25 unique 2-digit codes. Although we lose 
fine-grained detail by truncating the NAICS code, we aimed to optimize our
models by reducing variance introduced by high dimensionality. After applying
such dimension reductions, we eliminated extraneous variables, such as the 
Borrower's Zip Code and the Project's State, where were used to engineer
features. 

In addition to constructing features from the raw data, we also incorporated 
data from external sources, including monthly State-based measures of 
crime rate, GDP, and unemployment rate. We also joined in time-varying risk 
factors, including monthly snapshots of the `S&P 500`, `Consumer Price Index`, 
and 14 other volatility market indices. 

- BEN: Fill in where the data came from and any other important info 

###Data Splitting 

We randomly partitioned the data into 70% training and 30% test sets. 
This approach does not implement a time-based split, but rather a random 
sampling of observations over the entire 1990-2014 window. We adopted this 
splitting approach because we were interested in capturing the signal 
of the Great Recession within our models. Further, we did not create a 
validation set because we performed feature selection and hyper-parameter
optimization using cross-validation on the training set. 

###Data Preprocessing

After engineering features and joining in external data sources, 
we applied several preprocessing steps to our main data frame.
First, we centered and scaled the continuous predictors to apply regularization 
techniques during the modeling phase. Doing so adjusted for variables being
on different scales; for example, `Gross Approval` varies in dollar amounts 
from \$30,000  to \$4,000,000, whereas `Term in Months` ranges from 1 to 389. 
Second, we applied a filter to remove features with near zero variance to 
eliminate predictors that do not offer meaningful signal. 





###Feature Selection

To perform feature selection, we used recursive feature elimination
with 10-fold cross-validation. This method uses random forests to iteratively
remove variables with low variable importance, as measured by mean increase 
in out-of-bag area-under-the-curve (AUC). In other words, variables that 
do not contribute to significant improvements in AUC are eliminated. We
performed a grid search over the number of potential features to determine 
how many features to include. Note that factors were converted to separate dummy 
variables using a one-hot encoder. 



The following plot shows that recursive feature selection 
chose 122 
variables because AUC is maximized (see plot below). In effect, all variables
were kept because they offered predictive power regarding loan defaults. 

![](final_report_files/figure-html/unnamed-chunk-8-1.png)<!-- -->

The importances of the top 10 selected features are shown in the plot below.
We observe that State GDP, a monthly time-dependent risk factor, is the most 
important feature, meaning it led to the greatest average increase in AUC
across cross-validation iterations. State unemployment rate and crime rate 
are also highly important, suggesting local time-dependent risk factors 
are the most predictive of whether a loan defaults. 

The importance of NAICS code 72, corresponding to 
"Accommodation & Food Services", is consistent with our exploratory 
data analysis finding that the sector is especially risk prone. Borrower States
such as Michigan, California, and Florida also offer predictive power regarding 
defaulting. Lastly, the importances of the Collar Index (CLL) and Iron 
Butterfly Index (BFLY) imply market volatility measures also improve 
the discrimination of loan defaults. 

![](final_report_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

###Model Fitting 

Using these selected features, we fit models predicting the binary outcome
of whether a small business defaults on a loan. We constructed linear and 
nonlinear models, including a logistic regression model with the elastic net 
penalty, a random forest classifier, and a gradient boosting machine 
classifier. To tune hyper-parameters, we used 10-fold cross-validation with the 
one standard-error rule, which selects parameters that obtain the highest 
cross-validated AUC within one standard error of the maximum. For each model
type, we performed a grid search over the hyper-parameters to ensure optimal
selection.

####Logistic Regression with Elastic Net

First, AUC was used to select the optimal logistic regression model with an
elastic net penalty using the one standard-error rule. As shown on the plot 
below, the final values used for the model were `alpha` = 0.1 and `lambda` = 0,
indicated by the spike in the red curve at `alpha` = 0.1. This implies the 
optimal model used the ridge penalty more than the LASSO penalty with minimal 
regularization. 

![](final_report_files/figure-html/unnamed-chunk-10-1.png)<!-- -->

####Random Forest Classifier

Second, AUC was used to select the optimal random forest model, which selected 
`mtry` = 8 as the best parameter. This means 8 random predictors were chosen 
to build each tree of the random forest. The plot below shows steadily declining
AUC as the number of randomly chosen predictors increases, indicating that
the optimal model is sparsest. 

![](final_report_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

####Gradient Boosting Machine Classifier

Third, AUC was similarly used to select the optimal gradient boosting machine 
(GBM) model. The final values used for the model were `nrounds` = 100, 
`max_depth` = 6, `eta` = 0.03, `gamma` = 0, `colsample_bytree` = 0.4, 
`min_child_weight` = 1 and `subsample` = 0.5. This means that the tuning 
procedure utilized a learning rate of 0.03 and a minimum loss reduction of 0, 
resulting in the optimal model with 100 trees of maximum depth 6 that subsamples
50% of the observations and 40% of the features for each tree. This combination
of optimal hyper-parameters in shown by the spike of the red curve in the first 
subplot at the maximum tree depth of 6.  

![](final_report_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

###Model Evaluation 

After we optimized the hyper-parameters of our models, we evaluated the models  
using in-sample and out-of-sample metrics, including AUC, sensitivity,
ROC curves, and calibration. To do so, we used these "best" models to 
predicts loan defaults in the training and test sets. 

####In-Sample Evaluation 

#####Training AUC and Sensitivity of Best Models 

The following plot compares averaged **training** area under the ROC curve 
and sensitivity across the model types with optimized parameters. We observe
that the gradient boosting machine classifier has the highest AUC and 
sensitivity, whereas the logistic regression model with the elastic net penalty
performs the worst. 

![](final_report_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

#####Distribution of Resampled Training AUC, Sensitivity, and Specificity 

To examine the spread of **training** area under the ROC curve,
sensitivity, and specificity across model types, we leverage the resampled
data generated during the cross-validation of modeling fitting to plot 
their respective distributions. In the following plot, we observe that 
the GBM classifier has the highest median AUC, sensitivity, and specificity,
as well as the smallest spread. Although the random forest classifier has 
comparable sensitivity, it exhibits enormous variance compared to the other 
models, suggesting it is prone to overfitting. For this reason, the logistic
regression classifier (a linear model) outperforms the random forest classifier
(a non-linear model) in terms of AUC and specificity. 

![](final_report_files/figure-html/unnamed-chunk-14-1.png)<!-- -->

#####Training ROC Curves 

Lastly, we can examine the training ROC curves by model type. 
We observe that the random forest model has a near-perfect ROC curve,
which also implies it is overfitting to the training data. The GBM model again
performs worse than the random forest model on the training data, but likely 
because it is avoiding overfitting. The logistic regression model with the 
elastic net penalty performs the worst. 

![](final_report_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

####Out-of-Sample Evaluation 

#####Test ROC Curves 

We evaluated our best models on a held-out test set representing 30% of the 
original data. The ROC curves below reveal that the GBM model performed 
the best on the test set, followed by the logistic regression model, and 
finally, the random forest classifier. The weak performance of the random forest 
classifier is likely due to overfitting on the training set. Nevertheless, 
the all models achieve good performance over "random guessing" baselines. 



![](final_report_files/figure-html/unnamed-chunk-17-1.png)<!-- -->

#####Test Calibration Plots

Lastly, we evaluated the calibration of the our models' predicted probabilities
of loan default against the observed fraction of defaults in the data. 
A point on the dashed line means that the model's predicted probability
of default matched the empirical default rate. Points to the right of the line 
mean the model overestimated the default probability, whereas points to the 
left mean the model underestimated the default probability. 

The GBM model achieves the best calibration because its points follow 
the dashed line most closely. The logistic regression model with the elastic net
penalty achieves comparable performance; however, the random forest classifier
tends to overestimate default probabilities. Again, this weaker performance 
is likely due to overfitting. 

![](final_report_files/figure-html/unnamed-chunk-18-1.png)<!-- -->

The overfitting of the random forest classifier may be due to the fact that 
too many features were randomly selected to build trees at each
iteration. Our hyper-parameter optimization approach performed a grid search
over possible values of `mtry`, representing the number of features 
randomly chosen to build each tree in the forest. However, our grid may have
not been large enough, since the minimum value of `mtry` was chosen. 
However, computational resources limited our ability to refit models 
over a larger search space. 

Moreover, the gradient boosting machine classifier demonstrated the best
performance on the test set in terms of AUC and calibration.  

##Cox Proportional Hazards Models 

- THEO

#Modeling Loss at Default 

##Value-at-Risk

- ALEX

##Average Value-at-Risk

- ALEX

#Loss Distributions by Tranche

- BEN
