# This script selects 500 random loans from the test set and writes them to RDS. 

# Initialize libraries 
library(tidyverse)

# Initialize files 
test_file_in = "../data/test.rds"
portfolio_file_out = "../data/portfolio.rds"

# Read in test file 
test = read_rds(test_file_in)

# Extract 500 random loans from test set
set.seed(123)
portfolio = sample_n(test, size = 500)

# Write output file 
write_rds(portfolio, portfolio_file_out)