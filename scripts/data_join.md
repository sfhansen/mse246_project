MSE 246 Data Join
================
Samuel Hansen
1/21/2017

This script joins together data from the Small Business Association (SBA), S&P 500, State-level GDP, and State-level unemployment rates.

``` r
# Initialize libraries and input files 
library(knitr)
library(lubridate)
library(stringr)
library(zoo)
library(tidyverse)
loans_file_in <- "../data/SBA_Loan_data_full_edited.csv"
sp500_file_in <- "../data/SP500_ret.csv"
gdp_file_in <- "../data/STATE_GDP.csv"
unemploy_file_in <- "../data/unemployment_rates.csv"
cpi_file_in <- "../data/cpi.csv"
crime_file_in <- "../data/crime_rate.csv"
volatility_file_in <- "../data/volatility_markets.csv"
```

``` r
df <- 
  # Read in loan data 
  read_csv(loans_file_in) %>%
  # Rename variable
  plyr::rename(replace = c("2DigitNAICS" = "NAICS")) %>%
  # Convert year to date-time object
  mutate(ApprovalDate = mdy(ApprovalDate),
         month = month(ApprovalDate),
         year = year(ApprovalDate)) %>%
  
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
              select(year = Year, mean_cpi = Avg),
            by = c("year")) %>%
  
  # Join crime rate data 
  left_join(read_csv(crime_file_in) %>%
              plyr::rename(replace = c("Year" = "year")) %>%
              gather(BorrState, crime_rate, AL:WY),
            by = c("BorrState", "year")) %>%
  
  #  # Join volatility market data (file needs to be cleaned)
  # left_join(read_csv(volatility_file_in) %>%
  #             plyr::rename(replace = c("Date" = "date")) %>%
  #             mutate(date = as.yearmon(date, format = "%b-%y"),
  #                    month = month(date),
  #                    year = year(date)) %>%
  #             select(-date),
  #           by = c("year", "month")) %>%
  
  # Remove approval year and month as variables 
  select(-c(month, year))

# Write data frame to file
write_csv(df, "../data/final_merged.csv")
```
