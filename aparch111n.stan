// AGARCH(1,1,1) model for time series of returns
data {
  int<lower=1> N;             // Number of observations
  vector[N] y;                // returns (not de-meaned)
  real u0;                    // Initial error
  real<lower=0> h0;           // Initial variance
  
  real mu0_mu;                // Mean for Normal prior for mu
  real<lower=0> s0_mu;        // Standard deviation for normal prior for mu
  vector[5] mu0_gp;           // Mean for Normal prior for omega, alpha, and beta
  vector<lower=0>[5] s0_gp;   // Standard deviation for normal prior omega, alpha, and beta
  int<lower=0> n_steps_ahead; // Number of forecast steps ahead
}

parameters {
  real mu; // Mean
  real<lower=0> omega; // GARCH constant
  real<lower=0,upper=1> alpha;  // GARCH parameter for lagged errors
  real<lower=-1,upper=1> gamma; //Leverage parameter
  real<lower=0,upper=1> beta;    // GARCH parameter for lagged variances
  real<lower=0> delta; //volatility power parameter
}

transformed parameters {
  real st; //delta:th power of volatility (needs to be converted back to variance or volatility/stdev squared)
  vector[N] h;
  vector[N] lpdf;

  // Variance
  st = omega + alpha * pow(abs(u0) - gamma * u0, delta) + beta * pow(h0, delta/2); //s0 = vol0^d, vol0 = var0^0.5 or vol^2 = var0, var_t = s_t^{2/d}
  h[1] = pow(st, 2/delta);
  for (t in 2:N) {
    {
    real ut = y[t-1] - mu;
    st = omega + alpha * pow(abs(ut) - gamma * ut, delta) + beta * st;
    h[t] = pow(st, 2/delta);
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
  gamma ~ normal(mu0_gp[3], s0_gp[3]);
  beta ~ normal(mu0_gp[4], s0_gp[4]);
  delta ~ normal(mu0_gp[5], s0_gp[5]);
}

generated quantities {
  
  // forecasting
  vector[n_steps_ahead] y_fore;
  vector[n_steps_ahead] h_fore;
  vector[n_steps_ahead] lpdf_fore;
  real sft;
  
  {
  real ut = y[N] - mu;
  sft = omega + alpha * pow(abs(ut) - gamma * ut, delta) + beta * st;
  h_fore[1] = pow(sft, 2/delta);
  }
  y_fore[1] = normal_rng(mu, sqrt(h_fore[1] + 1e-8)); //mistake here corrected: h[N] -> h_fore[1]
  for (t in 2:n_steps_ahead) {
    {
    real ut = y_fore[t-1] - mu;
    sft = omega + alpha * pow(abs(ut) - gamma * ut, delta) + beta * sft;
    h_fore[t] = pow(sft, 2/delta);
    }
    y_fore[t] = normal_rng(mu, sqrt(h_fore[t] + 1e-8));
  }

  lpdf_fore = (y_fore - mu);
  lpdf_fore .*= (y_fore - mu);
  lpdf_fore = - 0.5*log(2*pi()) - 0.5*log(h_fore + 1e-8) - 0.5 * lpdf_fore ./ (h_fore + 1e-8);
}
