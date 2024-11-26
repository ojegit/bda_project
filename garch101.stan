// GARCH(1,0,1) model for time series of returns
data {
  int<lower=1> N;             // Number of observations
  vector[N] y;                // returns (not de-meaned)
  real u0;                    // Initial error
  real<lower=0> h0;           // Initial variance
  
  real mu0_mu;                // Mean for Normal prior for mu
  real<lower=0> s0_mu;        // Standard deviation for normal prior for mu
  vector[3] mu0_gp;           // Mean for Normal prior for omega, alpha, and beta
  vector<lower=0>[3] s0_gp;   // Standard deviation for normal prior omega, alpha, and beta
  
  int<lower=0> n_steps_ahead; // Number of forecast steps ahead
}

parameters {
  real mu; // Mean
  real<lower=0> omega;   // GARCH constant
  real<lower=0,upper=1> alpha;  // GARCH parameter for lagged errors
  //real<lower=0,upper=(1-alpha)> beta;    // GARCH parameter for lagged variances
  real<lower=0,upper=1> beta;    // GARCH parameter for lagged variances
}

transformed parameters {
  vector[N] h;
  vector[N] lpdf;

  // Variance
  h[1] = omega + alpha * u0 * u0 + beta * h0;
  for (t in 2:N) {
    {
    real ut = y[t-1] - mu;
    h[t] = omega + alpha * ut * ut + beta * h[t-1];
    }
  }
  
  lpdf = (y - mu);
  lpdf .*= (y - mu);
  lpdf = - 0.5*log(2*pi()) - 0.5*log(h + 1e-8) - 0.5 * lpdf ./ (h + 1e-8);
}


model {

  // Likelihood
  target += sum(lpdf);
  
  // Priors
  mu ~ normal(mu0_mu, s0_gp);
  omega ~ normal(mu0_gp[1], s0_gp[1]);
  alpha ~ normal(mu0_gp[2], s0_gp[2]);
  beta ~ normal(mu0_gp[3], s0_gp[3]);
}

generated quantities {
  
  // forecasting
  vector[n_steps_ahead] y_fore;
  vector[n_steps_ahead] h_fore;
  vector[n_steps_ahead] lpdf_fore;
  
  {
  real ut = y[N] - mu;
  h_fore[1] = omega + alpha * ut * ut + beta * h[N];
  }
  y_fore[1] = normal_rng(mu, sqrt(h[N] + 1e-8));
  for (t in 2:n_steps_ahead) {
    {
    real ut = y_fore[t-1] - mu;
    h_fore[t] = omega + alpha * ut * ut + beta * h_fore[t-1];
    }
    y_fore[t] = normal_rng(mu, sqrt(h_fore[t] + 1e-8));
  }

  lpdf_fore = (y_fore - mu);
  lpdf_fore .*= (y_fore - mu);
  lpdf_fore = - 0.5*log(2*pi()) - 0.5*log(h_fore + 1e-8) - 0.5 * lpdf_fore ./ (h_fore + 1e-8);
}
