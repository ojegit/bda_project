

### EXAMPLE FORECASTING FOR GARCH(1,0,1)

### load libraries
library(cmdstanr)
source("fit_and_predict_model.R")

### model settings
model_name <- "garch101n" #full model name including the dist i.e garch101n, msgarch101t etc
compile <- FALSE 
save_draws <- FALSE #this takes a long time the larger the results (.RDS or the fitted model is always saved)

# stan data inputs
model_data <- list(
  N = NA, #set at the forecast loop
  y = NA, #set at the forecast loop
  u0 = 0, #set at the forecast loop
  h0 = NA, #set at the forecast loop
  mu0_mu = rep(0,1),
  s0_mu = rep(1,1),
  mu0_gp = rep(0,3),
  s0_gp = rep(1,3),
  n_steps_ahead = 1
)


# open the stan file
if (!compile) {
  compiled_model <- cmdstan_model( exe_file = paste("./",model_name,".exe", sep="") )
} else {
  compiled_model <- cmdstan_model( paste("./",model_name,".stan", sep="") )
}

### load data
data <- read.csv("./data/m-geln.txt",header = FALSE) #General Electric stock monthly log-returns (https://faculty.chicagobooth.edu/ruey-s-tsay/research/analysis-of-financial-time-series-3rd-edition)
y <- data$V1
# data <- read.csv('./data/sp500_mo.txt', header = TRUE, sep = "\t")
# y <- 100*diff(log(data$SP500))
N <- length(y)


### create folders for results
save_folder <- file.path(".","results",model_name) #default: ./results/[model_name]/
dir.create(save_folder, recursive = TRUE) #comment out if path exists (it's checked anyway)


### forecasting settings
verbose <- 1
fore_win <- 10 #size of the forecasting window ( fore_win + fit_win = length(y) )
fit_win <- 800 #size of the fit window ( fore_win + fi_twin = length(y) )

if (fore_win + fit_win > N) {
  stop("fore_win + fit_win exceed full data length!")
}

### simulation settings
simulation_settings <- list()
simulation_settings$chains <- 2 #note: with fit_method = "hmc" the data is still saved into one file, with fit_method = "laplace" separate files are produced
simulation_settings$save_folder <- save_folder
simulation_settings$basename <- NA
simulation_settings$iter_sampling <- 2000
simulation_settings$show_messages <- FALSE
simulation_settings$fit_method <- "laplace" #other options: "hmc"
simulation_settings$model_data <- model_data
simulation_settings$save_draws <- save_draws


# forecast loop
for (i in 1:fore_win) { 
  
  #get a subset of the data
  fit_start <- 1 + (i - 1)
  fit_end <-  fit_win + (i - 1)
  y_slice <- y[fit_start : fit_end]
  
  if (verbose > 0) {
    cat(paste("FORE_WIN: ",i," / ",fore_win, " ( ",100*i/fore_win,"% ) | FIT_START: ",fit_start,", FIT_END: ",fit_end,"\n", sep=""))
  }
  
  #update model_data for the new fit window
  simulation_settings$model_data$y <- y_slice #don't demean since mu is estimated!
  simulation_settings$model_data$N <- length( y_slice )
  simulation_settings$model_data$u0 <- 0
  simulation_settings$model_data$h0 <- mean( (y - mean(y))^2 )
  simulation_settings$basename <- paste("fore_win_",i,sep = "")
  
  # fit and forecast
  fit_and_predict_model(compiled_model, simulation_settings)
}
### EOF