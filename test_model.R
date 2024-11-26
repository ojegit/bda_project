
### load libraries
library(cmdstanr)
library(ggplot2)
library(bayesplot)
library(posterior)
library(loo)
#library(bridgesampling)


### load data
data <- read.csv("./Data/m-geln.txt",header = FALSE) #Gemeral Electric stock monthly log-returns (https://faculty.chicagobooth.edu/ruey-s-tsay/research/analysis-of-financial-time-series-3rd-edition)
y <- data$V1
N <- length(y)


### model settings
# model name
model_name <- "msgarch101" #available models: garch101, garch111, egarch101, egarch111, msgarch101, msgarch111


# simulation options
refresh = 10
iter_warmup = 2000
iter_sampling = 1000
chains = 2
parallel_chains = 2
show_messages = TRUE

# compile stan file
compile <- FALSE #set to FALSE if already compiled and the .exe file is loaded instead (modify for Linux)

# generate file names
model_file <- paste("./",model_name,".stan", sep="")
compiled_file <- paste("./",model_name,".exe", sep="")


if (model_name == "garch101") {
  
  par_names_fit <- c("mu", "omega", "alpha", "beta")
  par_names_fore <- c("y_fore[1]", "h_fore[1]",'lpdf_fore[1]')
  
  model_data <- list(
    N = N,
    y = y,
    u0 = 0,
    h0 = mean( (y-mean(y))^2 ),
    mu0_mu = rep(0,1),
    s0_mu = rep(1,1),
    mu0_gp = rep(0,3),
    s0_gp = rep(1,3),
    n_steps_ahead = 1
  )
  
  
} else if(model_name == "garch111") {
  
  par_names_fit <- c("mu", "omega", "alpha1", "alpha2", "beta")
  par_names_fore <- c("y_fore[1]", "h_fore[1]",'lpdf_fore[1]')
  
  model_data <- list(
    N = N,
    y = y,
    u0 = 0,
    h0 = mean( (y-mean(y))^2 ),
    mu0_mu = rep(0,1),
    s0_mu = rep(1,1),
    mu0_gp = rep(0,4),
    s0_gp = rep(1,4),
    n_steps_ahead = 1
  )
  
} else if(model_name == "egarch101") {
  
  par_names_fit <- c("mu", "omega", "alpha", "beta")
  par_names_fore <- c("y_fore[1]", "x_fore[1]",'lpdf_fore[1]')
  
  model_data <- list(
    N = N,
    y = y,
    u0 = 0,
    x0 = log( mean( (y-mean(y))^2 ) ),
    mu0_mu = rep(0,1),
    s0_mu = rep(1,1),
    mu0_gp = rep(0,3),
    s0_gp = rep(1,3),
    n_steps_ahead = 1
  )
  
  
} else if(model_name == "egarch111") {
  
  par_names_fit <- c("mu", "omega", "alpha1", "alpha2", "beta")
  par_names_fore <- c("y_fore[1]", "x_fore[1]",'lpdf_fore[1]')
  
  model_data <- list(
    N = N,
    y = y,
    u0 = 0,
    x0 = log( mean( (y-mean(y))^2 ) ),
    mu0_mu = rep(0,1),
    s0_mu = rep(1,1),
    mu0_gp = rep(0,4),
    s0_gp = rep(1,4),
    n_steps_ahead = 1
  )
  
} else if(model_name == "msgarch101") {
  
  par_names_fit <- c("mu", "omega1", "omega2", "alpha1", "alpha2", "beta1", "beta2", "p11", "p22")
  par_names_fore <- c("y_fore[1]", "h_fore[1]",'lpdf_fore[1]')
  
  model_data <- list(
    N = N,
    y = y,
    u0 = 0,
    h0 = mean( (y-mean(y))^2 ),
    mu0_mu = rep(0,1),
    s0_mu = rep(1,1),
    mu0_gp = rep(0,6),
    s0_gp = rep(1,6),
    a0_tp = rep(1,2),
    b0_tp = rep(2,2),
    n_steps_ahead = 1
  )
  
  
} else if(model_name == "msgarch111") {
  
  par_names_fit <- c("mu", "omega1", "omega2", "alpha11", "alpha12", "alpha21", "alpha22", "beta1", "beta2", "p11", "p22") 
  par_names_fore <- c("y_fore[1]", "h_fore[1]",'lpdf_fore[1]')
  
  model_data <- list(
    N = N,
    y = y,
    u0 = 0,
    h0 = mean( (y-mean(y))^2 ),
    mu0_mu = rep(0,1),
    s0_mu = rep(1,1),
    mu0_gp = rep(0,8),
    s0_gp = rep(1,8),
    a0_tp = rep(1,2),
    b0_tp = rep(2,2),
    n_steps_ahead = 1
  )
  
} else {
  
  stop("Model name not found!")
  
}

# open the stan file
if (!compile) {
  compiled_model <- cmdstan_model(exe_file = compiled_file)
} else {
  compiled_model <- cmdstan_model(model_file)
}


### fit model
model_fit <- compiled_model$sample(
  data = model_data,
  refresh = refresh,
  iter_warmup = iter_warmup,
  iter_sampling = iter_sampling,
  chains = chains,
  parallel_chains =  parallel_chains,
  show_messages = show_messages
)




# plot fit pars' distributions
mcmc_hist(model_fit$draws(format = 'draws_list'), pars = par_names_fit)

# plot fit pars' line plots
mcmc_trace(model_fit$draws(format = 'draws_list'), pars = par_names_fit) +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73"))


# plot fit model stats
model_fit$summary(variables = par_names_fit)

summarise_draws(model_fit$draws(format = 'draws_list', variables = par_names_fit), 
                Rhat=rhat_basic, ESS= ess_mean, ~ess_quantile(.x, probs = 0.05))


# print information criteria
waic_fit_out <- waic(model_fit$draws("lpdf"))
loo_fit_out <- loo(model_fit$draws("lpdf"))
print(waic_fit_out)
print(loo_fit_out)


# marginal likelihood
#bs_out <- bridge_sampler(model_fit)
#lml_out <-bs_out$log_marginal_likelihood
#print(lml_out)


### predict model
model_pred <- compiled_model$generate_quantities(
  fitted_params = model_fit,
  data = model_data,
  parallel_chains = parallel_chains
)

# plot predictions' distributions
mcmc_hist(model_pred$draws(), pars=par_names_fore)


# plot predictions' line plots
mcmc_trace(model_pred$draws(), pars=par_names_fore) +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73"))


# print predicted model stats
model_pred$summary(variables = par_names_fore)

summarise_draws(model_pred$draws(format = 'draws_list', variables = par_names_fore), 
                Rhat=rhat_basic, ESS= ess_mean, ~ess_quantile(.x, probs = 0.05))


### EOF