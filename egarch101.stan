// EGARCH(1,0,1) model or EGARCH(1,1) without a leverage term for time series of returns

data {
  int<lower=1> N;             // Number of observations
  vector[N] y;                // returns (not de-meaned)
  real u0;                    // Initial error
  real x0;                    // Initial log-variance
  real mu0_mu;                // Mean for Normal prior for mu
  real<lower=0> s0_mu;        // Stdev for normal prior for mu
  vector[3] mu0_gp;           // Mean for Normal prior for omega, alpha, and beta
  vector<lower=0>[3] s0_gp;   // Stdev for normal prior for omega, alpha, and beta
  int<lower=0> n_steps_ahead; // Number of forecast steps ahead
}

// Parameters
parameters {
  real mu; // Mean
  real omega;   // Constant
  real<lower=-1,upper=1> alpha;  // ARCH parameter
  real<lower=-1,upper=1> beta;    // GARCH parameter
}

transformed parameters {
  vector[N] lpdf; //log-density
  vector[N] x;    //log-variances
  
  // log-variance
  {
  real z = u0 * exp(-0.5*x0);
  x[1] = omega + alpha * (abs(z) - 0.7979) + beta * x0;
  }
  for (t in 2:N) {
    {
    real z = (y[t-1] - mu) * exp(-0.5*x[t-1]);
    x[t] = omega + alpha * (abs(z) - 0.7979) + beta * x[t-1];
    }
  }
  
  lpdf = (y - mu);
  lpdf .*= (y - mu);
  lpdf = - 0.5*log(2*pi()) - 0.5*x - 0.5 * lpdf .* exp(-x);
}

// Model block
model {

  // Likelihood
  target += sum(lpdf);
  
  // Priors
  mu ~ normal(mu0_mu, s0_mu);
  omega ~ normal(mu0_gp[1], s0_gp[1]);
  alpha ~ normal(mu0_gp[2], s0_gp[2]);
  beta ~ normal(mu0_gp[3], s0_gp[3]);
  
}

generated quantities {
  
  // forecasting
  vector[n_steps_ahead] y_fore; // Forecasted returns
  vector[n_steps_ahead] x_fore; // Forecasted log-variances
  vector[n_steps_ahead] lpdf_fore;
  
  {
  real z = (y[N] - mu) * exp(-0.5*x[N]);
  x_fore[1] = omega + alpha * (abs(z) - 0.7979) + beta * x[N]; // Forecast for the first step
  }
  y_fore[1] = normal_rng(mu, exp(-0.5*x_fore[1]) + 1e-8);  // Forecast return
  for (t in 2:n_steps_ahead) {
    {
    real z = (y_fore[t-1] - mu) * exp(-0.5*x_fore[t-1]);
    x_fore[t] = omega + alpha * (abs(z) - 0.7979) + beta * x_fore[t-1];  // Update forecast variance
    }
    y_fore[t] = normal_rng(mu, exp(-0.5*x_fore[t]) + 1e-8);  // Forecast return
  }
  
  lpdf_fore = (y_fore - mu);
  lpdf_fore .*= (y_fore - mu);
  lpdf_fore = - 0.5*log(2*pi()) - 0.5*x_fore - 0.5 * lpdf_fore .* exp(-x_fore);
}
