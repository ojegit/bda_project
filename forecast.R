### forecasting

### load data

fore_win <- 100
fit_win <- 800

# forecast loop
for (i = 1:fore_win) {

    #get subset of the data
    fit_start <- 1 + (i - 1)
    fit_end <-  fit_win + (i - 1)

    y_slice <- y[fit_start : fit_end]

    ### 1. FIT MODEL 
    ### 2. FORECAST MODEL
    ### 3. SAVE RESULTS
}