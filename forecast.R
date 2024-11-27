### forecasting

### load data



### forecast settings
iter_warmup <- 2000
iter_sampling <- 1000
chains <- 2
parallel_chains <- 2

n_steps_ahead <- 1
save_name <- "model_name"
fore_win <- 100
fit_win <- 800

# forecast loop

#https://nceas.github.io/oss-lessons/parallel-computing-in-r/parallel-computing-in-r.html
#https://www.blasbenito.com/post/02_parallelizing_loops_with_r/
#https://sparkbyexamples.com/r-programming/run-r-for-loop-in-parallel/

for (i in 1:fore_win) { 

    #get a subset of the data
    fit_start <- 1 + (i - 1)
    fit_end <-  fit_win + (i - 1)

    y_slice <- df$y[fit_start : fit_end]

    ### 1. FIT MODEL 
    ### 2. FORECAST MODEL
    ### 3. SAVE FIT AND PRED MODEL RESULTS
    #https://mc-stan.org/cmdstanr/reference/fit-method-save_output_files.html
    #https://discourse.mc-stan.org/t/import-csv-output-from-cmdstan-in-r-how-to-indicate-the-chain/20595/3
    #paste(save_name,'_fore_step_',i,sep="")
}