# mse246_project
This is the repository for MS&amp;E 246: Financial Risk Analytics Project.

This repository is broken into 4 sections:

- `reports`: includes all code, text, and figures for the final report;
- `scripts`: includes R scripts that perform specific tasks (i.e. data joining);
- `studies`: includes exploratory analyses and model building pipelines;
- `data`: includes all data frames, including raw SBA data and external sources. 

For viewability, see files ending in `.md`, which show plots (when available). Otherwise, see `.Rmd` files to view all code. 

The most important files to note are: 

- `data_summary.md`: performs exploratory data analysis;
- `data_join.md`: performs data cleaning and joins raw data with external sources;
- `model_fitting.md`: fits binary response models of loan default;
- `loss_at_default_model.md`: fits loss at default model and estimates VaR and AVaR; 
- `cox_processing.R`, `cox_models.R`, `cox_diagnostic_functions.R`, `cox_refit_best_model.R`, and `cox_surv_probs.R`: taken together, fit Cox proportional hazards model and perform analysis. 
- `final_report.Rmd`: contains all remaining analysis, including Tranche distribution estimation. 
