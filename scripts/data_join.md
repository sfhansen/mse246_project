MSE 246 Data Join
================
Samuel Hansen
1/21/2017

-   [Data Join](#data-join)
-   [Data Splitting](#data-splitting)

Data Join
=========

This script joins together data from the Small Business Association (SBA), S&P 500, State-level GDP, and State-level unemployment rates.

``` r
# Initialize libraries 
library(knitr)
library(lubridate)
library(stringr)
library(zoo)
library(tidyverse)

# Initialize input files 
loans_file_in <- "../data/SBA_Loan_data_full_edited.csv"
sp500_file_in <- "../data/SP500_ret.csv"
gdp_file_in <- "../data/STATE_GDP.csv"
unemploy_file_in <- "../data/unemployment_rates.csv"
cpi_file_in <- "../data/cpi.csv"
crime_file_in <- "../data/crime_rate.csv"
volatility_file_in <- "../data/volatility_markets.csv"

# Initialize output files 
out_file <- "../data/merged.rds"
train_out_file <- "../data/train.rds"
test_out_file <- "../data/train.rds"
```

``` r
df <- 
  
  # Read in loan data 
  read_csv(loans_file_in) %>%
  plyr::rename(replace = c("2DigitNAICS" = "NAICS")) %>%
  mutate(ApprovalDate = mdy(ApprovalDate),
         month = month(ApprovalDate),
         year = year(ApprovalDate)) %>%
  
   # Remove "Exempt" Loans
  filter(LoanStatus != "EXEMPT") %>%
  
  # Join S&P 500 data 
  left_join(read_csv(sp500_file_in) %>%
              mutate(Date = mdy(Date),
                     month = month(Date),
                     year = year(Date)) %>% 
              select(-Date), 
            by = c("month", "year")) %>%
  
  # Join State GDP data 
  left_join(read_csv(gdp_file_in) %>%
              plyr::rename(replace = c("STATE" = "year")) %>%
              gather(BorrState, gdp, AL:WY), 
              by = c("BorrState", "year")) %>%
  
  # Join unemployment rate data 
  left_join(read_csv(unemploy_file_in) %>%
              plyr::rename(replace = c("Month" = "date")) %>%
              mutate(date = as.yearmon(date, format = "%b-%y"),
                     month = month(date),
                     year = year(date)) %>%
              select(-date) %>%
              gather(BorrState, unemploy_rate, AL:WY), 
            by = c("month", "year", "BorrState")) %>%
  
  # Join consumer price index data 
  left_join(read_csv(cpi_file_in) %>%
              plyr::rename(replace = c("DATE" = "date",
                                       "VALUE" = "cpi")) %>%
              mutate(date = mdy(date),
                     month = month(date),
                     year = year(date)) %>%
              select(-date),
            by = c("year", "month")) %>%
  
  # Join crime rate data 
  left_join(read_csv(crime_file_in) %>%
              plyr::rename(replace = c("Year" = "year")) %>%
              gather(BorrState, crime_rate, AL:WY),
            by = c("BorrState", "year")) %>%
  
   # Join volatility market data 
  left_join(read_csv(volatility_file_in) %>%
              plyr::rename(replace = c("Date" = "date")) %>%
              mutate(date = mdy(date),
                     month = month(date),
                     year = year(date)) %>%
              select(-date),
            by = c("year", "month")) %>%
  
  # Recode NA values 
  dmap_at('ChargeOffDate', ~ifelse(.x == 'N/A', NA, .x)) %>%
  
  mutate(
    # Convert Charge off date to date-time object 
    ChargeOffDate = mdy(ChargeOffDate),
    
    # Extract first digit of zip code
    first_zip_digit = str_sub(BorrZip, end = 1), 
         
    # Encode rare subprograms as "Other"
    subpgmdesc = if_else(subpgmdesc == "Delta" | subpgmdesc == "Refinance",
                              "Other", subpgmdesc),
    
    # Re-encode Charge-off
    LoanStatus = if_else(LoanStatus == "CHGOFF", "default", "paid")) %>%
  
  # Remove unnecessary variables 
  select(-c(month, year, DeliveryMethod, TermByYear, BorrZip, 
            ProjectState, SameThirdPartyLendingState)) %>%
  
  # Convert character columns to factors
  dmap_if(is.character, as.factor) %>%
  
  # Remove remaining NAs 
  drop_na(-ChargeOffDate)

# Write data frame to file
write_rds(df, out_file)
```

Data Splitting
==============

We randomly partition the data into 70% training and 30% test sets. This approach does not implement a time-based split, but rather a random sampling of observations over the entire 1990-2014 window.

``` r
# Split data into train and test sets 
set.seed(1234)
percent_in_train <- 0.7
train_indices <- sample(nrow(df), size = percent_in_train*nrow(df))
train <- df[train_indices, ]
test <- df[-train_indices, ]

# Write data splits to files 
write_rds(train, train_out_file)
write_rds(test, test_out_file)
```
