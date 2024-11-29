################################################################
### fit_and_predict_model.R
################################################################


### load libraries
library(cmdstanr)
library(parallel)
library(doParallel)

fit_and_predict_model <- function(compiled_model, simulation_settings) {

  # warning: no input checks 
  
  fit_method <- simulation_settings$fit_method
  basename <- simulation_settings$basename
  save_folder <- simulation_settings$save_folder
  model_data <- simulation_settings$model_data
  show_messages <- simulation_settings$show_messages
  iter_sampling <- simulation_settings$iter_sampling
  chains <- simulation_settings$chains
  save_draws <- simulation_settings$save_draws

  
  #get model name
  model_name <- compiled_model$model_name()
  
  # open the compiled stan file
  compiled_model <- cmdstan_model(exe_file = paste(model_name, ".exe", sep=""))
  
  
  ### fit model
  if (fit_method == "laplace") {

    #fit pre model
    model_fit_pre <- compiled_model$optimize(
      algorithm = "lbfgs", #"lbfgs", "bfgs", or "newton"
      data = model_data, 
      show_messages = show_messages,
      jacobian = TRUE)
  
    
    if (chains > 1) {

      #fit model
      cl <- makeCluster(chains)
      registerDoParallel(cl)
      fit_list <- foreach(i = 1:chains) %dopar% {
        cat(paste("CHAIN NO: ",i,"/",chains,sep=""))
        model_fit <- compiled_model$laplace(
          data = model_data,
          mode = model_fit_pre,
          show_messages = show_messages,
          draws = iter_sampling)
      }
      
      #save results
      for (i in 1:chains) {
        if(save_draws) {
          df_tmp <- fit_list[[i]]$draws(format="df")
          write.csv(df_tmp, file = file.path(save_folder, paste(basename,"_draws_chain_",i,".csv",sep="")), row.names = FALSE)
        }
        fit_list[[i]]$save_object(file = file.path(save_folder, paste(basename,"_fitted_model_object_chain_",i,".RDS",sep="")))
      }
      
      stopCluster(cl)
      
    } else {
      
      #fit model
      model_fit <- compiled_model$laplace(
        data = model_data,
        mode = model_fit_pre,
        show_messages = show_messages,
        draws = iter_sampling)
    
      #save results
      if(save_draws) {
        df_tmp <- model_fit$draws(format="df")
        write.csv(df_tmp, file = file.path(save_folder, paste(basename,"_draws_chain_1.csv",sep="")), row.names = FALSE)
      }
      model_fit$save_object(file = file.path(save_folder, paste(basename,"_fitted_model_object_chain_1.RDS",sep="")))
        
    }
    

    
  } else if (fit_method == "hmc") {
    
    refresh <- simulation_settings$refresh
    iter_warmup <- simulation_settings$iter_warmup
    parallel_chains <- simulation_settings$parallel_chains
    
    #fit model
    model_fit <- compiled_model$sample(
      data = model_data,
      refresh = refresh,
      iter_warmup = iter_warmup,
      iter_sampling = iter_sampling,
      chains = chains,
      parallel_chains = parallel_chains,
      show_messages = show_messages
    )
    
    #save results
    if(save_draws) {
      df_tmp <- model_fit$draws(format="df")
      write.csv(df_tmp, file = file.path(save_folder, paste(basename,"_draws_chain_1.csv",sep="")), row.names = FALSE)
    }
    model_fit$save_object(file = file.path(save_folder, paste(basename,"_fitted_model_object_chain_1.RDS",sep="")))
  }

}
### EOF