##This script joins together data from the Small Business Association (SBA),
##S&P 500, State-level GDP, and State-level unemployment rates. 

## Initialize libraries and input files 
require(data.table)
require(magrittr)

merged_data_out <- '../data/SBA_Loan_data_full_edited_merged.csv'
loans_file_in <- "../data/SBA_Loan_data_full_edited.csv"
sp500_fioe_in <- "../data/SP500_ret.csv"
gdp_file_in <- "../data/STATE_GDP.csv"
unemploy_file_in <- "../data/unemployment_rates.csv"
crime_rate_in <- '../data/cpi.csv'
cpi_in <- '../data/crime_rate.csv'

#################################

sba_dt <- fread(loans_file_in)
sp500_dt <- fread(sp500_fioe_in)
gdp_dt <- fread(gdp_file_in)
unempl_dt <- fread(unemploy_file_in)
crime_dt <- fread(crime_rate_in)
cpi_dt <- fread(cpi_in)

sba_dt[, join_time := as.Date(ApprovalDate,"%m/%d/%Y")]
sp500_dt[, join_time := as.Date(Date,"%m/%d/%y")]
gdp_dt[, join_time := as.Date(as.character(STATE), "%Y")]
unempl_dt[, join_time := as.Date(Month, "%b-%y")]
crime_dt[, join_time := as.Date(as.character(Year), "%Y")]
cpi_dt[, join_time := as.Date(as.character(Year), "%Y")]

setkey(sba_dt, join_time)
setkey(sp500_dt, join_time)
setkey(gdp_dt, join_time)
setkey(unempl_dt, join_time)
setkey(crime_dt, join_time)
setkey(cpi_dt, join_time)

##keeps the interior dt join_time, eliminates the other one
merged_df <- sp500_dt[sba_dt, roll='nearest'] %>%
    .[, Date := NULL] %>%
    gdp_dt[., roll='nearest'] %>%
    .[, STATE := NULL] %>%
    unempl_dt[., roll='nearest'] %>%
    .[, Month := NULL] %>% 
    crime_dt[., roll='nearest'] %>%
    .[, Year := NULL] %>% 
    cpi_dt[., roll='nearest'] %>%
    .[, c("Year", "join_time") := NULL]

##Only date left will be ApprovalDate

write.csv(merged_df, merged_data_out)
