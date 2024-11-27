// GARCH(1,1,1) or GJR model for time series of returns
data {
  int<lower=1> N;             // Number of observations
  vector[N] y;                // returns (not de-meaned)
  real u0;                    // Initial error
  real<lower=0> h0;           // Initial variance
  // hyperparameters
  real mu0_mu;                // Mean for Normal prior for mu
  real<lower=0> s0_mu;        // Variance for normal prior for mu
  vector[4] mu0_gp;           // Mean for Normal prior for omega, alpha1, alpha2 and beta
  vector<lower=0>[4] s0_gp;   // Variance for normal prior omega, alphaa, alpha2 and beta
  real<lower=0> lambda0_nu;      // Rate parameter for exponential prior for student's t degrees of freedom
  
  int<lower=1> n_steps_ahead; // Number of forecast steps ahead
}


parameters {
  real mu; // Mean
  real<lower=0> omega;   // GARCH constant
  real<lower=0,upper=1> alpha1;  // ARCH parameter
  real<lower=0,upper=1> alpha2;  // Leverage parameter
  real<lower=0,upper=1> beta;    // GARCH parameter
  real<lower=2> nu; //degrees of freedom parameter for student's t distribution
}


transformed parameters {
  vector[N] lpdf; 
  vector<lower=0>[N] h;

  // Variance
  h[1] = omega + alpha1 * u0 * u0 + alpha2 * (u0 < 0) * u0 * u0 + beta * h0;
  for (t in 2:N) {
    {
    real ut = y[t-1] - mu;
    h[t] = omega + alpha1 * ut * ut + alpha2 * (ut < 0) * ut * ut + beta * h[t - 1];
    }
  }
  
  lpdf = (y - mu);
  lpdf .*= (y - mu);
  lpdf = lgamma( 0.5*(nu+1) ) -  lgamma( 0.5*nu ) - 0.5*log(pi()*(nu-2)) - 0.5*log( h + 1e-8 ) - 0.5*(nu+1) * log( 1 + lpdf ./ (h * (nu-2) + 1e-8) );
}

model {
  // Likelihood
  target += sum(lpdf);
  
  // Priors
  mu ~ normal(mu0_mu, s0_mu);
  omega ~ normal(mu0_gp[1], s0_gp[1]);
  alpha1 ~ normal(mu0_gp[2], s0_gp[2]);
  alpha2 ~ normal(mu0_gp[3], s0_gp[3]);
  beta ~ normal(mu0_gp[4], s0_gp[4]);
  nu ~ exponential(lambda0_nu);
}

generated quantities {
  
  // forecasting
  vector[n_steps_ahead] y_fore; // Forecasted returns
  vector[n_steps_ahead] h_fore; // Forecasted variances (note: can't use constraints here!)
  vector[n_steps_ahead] lpdf_fore; //lpdf pred

  {
  real ut = y[N] - mu;
  h_fore[1] = omega + alpha1 * ut * ut + alpha2 * (ut < 0) * ut * ut + beta * h[N]; // Forecast for the first step
  y_fore[1] = student_t_rng(nu, mu, sqrt(h_fore[1] + 1e-8));
  }
  for (t in 2:n_steps_ahead) {
    {
    real ut = y_fore[t-1] - mu;
    h_fore[t] = omega + alpha1 * ut * ut + alpha2 * (ut < 0) * ut * ut + beta * h_fore[t-1];  // Update forecast variance
    y_fore[t] = student_t_rng(nu, mu, sqrt(h_fore[t] + 1e-8));  // Forecast return
    }
  }
  
  lpdf_fore = (y_fore - mu);
  lpdf_fore .*= (y_fore - mu);
  lpdf_fore = lgamma( 0.5*(nu+1) ) - lgamma( 0.5*nu ) - 0.5*log( pi()*(nu-2) ) - 0.5*log( h_fore + 1e-8 ) - 0.5*(nu+1) * log( 1 + lpdf_fore ./ (h_fore * (nu-2) + 1e-8) );

}
