//MS-GARCH(1,1), Haas (2004) parameterization

data {
  int<lower=0> N; //number of observations
  vector[N] y; //returns
  real<lower=0> h0; //initial variance
  real u0; //initial error
  
  //hyperparameters
  real mu0_mu; //Mean, normal, mu
  real<lower=0> s0_mu; //Mean, normal, stdev
  vector[6] mu0_gp; //Variance, normal, mu
  vector<lower=0>[6] s0_gp; //Variance, normal, stdev 
  vector<lower=0>[2] a0_tp; //transition probabilities (tp) matrix diagonals, beta, shape
  vector<lower=0>[2] b0_tp; //transition probabilities (tp) matrix diagonals, beta, scale
  real<lower=0> lambda0_nu;      // Rate parameter for exponential prior for student's t degrees of freedom
  
  int<lower=1> n_steps_ahead; // Number of forecast steps ahead
}

parameters {
  real mu; //mean eq
  real<lower=0> omega1; //reg1 constant
  //real<lower=0> omega2; //reg2 constant
  real<lower=omega1> omega2; //reg2 constant (lower limit set to mitigate label switching)
  real<lower=0,upper=1> alpha1; //reg1 ARCH coeff
  real<lower=0,upper=1> alpha2; //reg2 ARCH coeff
  real<lower=0,upper=1> beta1; //reg1 GARCH coeff
  real<lower=0,upper=1> beta2; //reg2 GARCH coeff
  real<lower=0,upper=1> p11; //tp matrix element [1,1]
  real<lower=0,upper=1> p22; //tp matrix element [2,2]
  real<lower=2> nu; //degrees of freedom parameter for student's t distribution
}


transformed parameters {
  matrix[2,1] ht; //variance
  matrix[2,2] P; //tp matrix
  matrix[2,1] L; //likelihood (temporary)
  matrix[2,1] ppt; //posterior probability
  matrix[2,1] fpt; //filtered probability
  real c;
  real k = lgamma( 0.5*(nu+1) ) -  lgamma( 0.5*nu ) - 0.5*log(pi()*(nu-2)); //lpdf term
  real u1;
  matrix[2,1] u2;
  vector[N] lpdf; //log-density
  
  //storages
  array[N,2] real h;
  array[N,2] real pp;
  array[N+1,2] real fp;
  
  
  //tp matrix
  P[1,1] = p11;
  P[1,2] = (1-p11);
  P[2,1] = (1-p22);
  P[2,2] = p22;
  
  // Variance
  //t = 1
  ht[1,1] = omega1 + alpha1 * u0 * u0 + beta1 * h0; //reg 1
  ht[2,1] = omega2 + alpha2 * u0 * u0 + beta2 * h0; //reg 2
  
  fpt[1, 1] = (1-p22)/(2-p11-p22); //initialize at steady state prob
  fpt[2, 1] = 1 - fpt[1,1];
  fp[1, 1] = fpt[1, 1];   fp[1, 2] = fpt[2, 1];
  
  
  u2[1,1] = y[1] - mu;
  u2[2,1] = u2[1,1];
  
  L = log(fpt + 1e-8) + k - 0.5*log( ht + 1e-8 ) - 0.5*(nu+1) * log( 1 + (u2 .* u2) ./ (ht * (nu-2) + 1e-8) );
  c = max(L);
  
  L = exp(L - c);
  ppt = L/(L[1,1] + L[2,1]);
  fpt = P * ppt;
  
  lpdf[1] = c + log(L[1,1] + L[2,1]); //log-density (log-sum-exp)
  
  h[1, 1] = ht[1, 1];   h[1, 2] = ht[2, 1];
  pp[1, 1] = ppt[1, 1]; pp[1, 2] = ppt[2, 1];
  fp[2, 1] = fpt[1, 1]; fp[2, 2] = fpt[2, 1];
  
  //t > 1
  for (t in 2:N) {
    
    u1 = y[t-1] - mu;
    ht[1,1] = omega1 + alpha1 * u1 * u1 + beta1 * ht[1,1];
    ht[2,1] = omega2 + alpha2 * u1 * u1 + beta2 * ht[2,1];

    u2[1,1] = y[t] - mu;
    u2[2,1] = u2[1,1];
    
    L = log(fpt + 1e-8) + k - 0.5*log( ht + 1e-8 ) - 0.5*(nu+1) * log( 1 + (u2 .* u2) ./ (ht * (nu-2) + 1e-8) );
    c = max(L);
    
    L = exp(L - c);
    ppt = L/(L[1,1] + L[2,1]);
    fpt = P * ppt;
    
    lpdf[t] = c + log(L[1,1] + L[2,1]); //log-density (log-sum-exp)
    
    h[t, 1] = ht[1, 1];     h[t, 2] = ht[2, 1];
    pp[t, 1] = ppt[1, 1];   pp[t, 2] = ppt[2, 1];
    fp[t+1, 1] = fpt[1, 1]; fp[t+1, 2] = fpt[2, 1];
  }
  
}

model {

  // log-likelihood
  target += sum(lpdf);

  // Priors
  mu ~ normal(mu0_mu, s0_mu);
  omega1 ~ normal(mu0_gp[1],s0_gp[1]);
  omega2 ~ normal(mu0_gp[2],s0_gp[2]);
  alpha1 ~ normal(mu0_gp[3],s0_gp[3]);
  alpha2 ~ normal(mu0_gp[4],s0_gp[4]);
  beta1 ~ normal(mu0_gp[5],s0_gp[5]);
  beta2 ~ normal(mu0_gp[6],s0_gp[6]);
  p11 ~ beta(a0_tp[1], b0_tp[1]);
  p22 ~ beta(a0_tp[2], b0_tp[2]);
  nu ~ exponential(lambda0_nu);
}


generated quantities {
  
    // forecasting
    matrix[2,1] hft;
    matrix[2,1] ppft;
    matrix[2,1] yft;
    matrix[2,1] uft;
    matrix[2,1] Lf;
    real cf;
    vector[n_steps_ahead] y_fore; // Forecasted returns
    vector[n_steps_ahead] h_fore; // Forecasted lvariances
    vector[n_steps_ahead] lpdf_fore; //predictive lpdf
    
    // t = 1
    uft[1,1] = y[N] - mu;
    uft[2,1] = uft[1,1];
    
    hft[1,1] = omega1 + alpha1 * uft[1,1] * uft[1,1] + beta1 * h[N, 1]; //reg 1
    hft[2,1] = omega2 + alpha2 * uft[2,1] * uft[2,1] + beta2 * h[N, 2]; //reg 2
    
    ppft[1,1] = pp[N, 1];
    ppft[2,1] = pp[N, 2];
    
    ppft = P*ppft;
    
    
    yft[1,1] = student_t_rng(nu, mu, sqrt(hft[1,1] + 1e-8) );
    yft[2,1] = student_t_rng(nu, mu, sqrt(hft[2,1] + 1e-8) ); 
    
    y_fore[1] = yft[1,1]*ppft[1,1] + yft[2,1]*ppft[2,1]; //collapse regs
    h_fore[1] = hft[1,1]*ppft[1,1] + hft[2,1]*ppft[2,1]; //collapse regs
    
    uft = yft - mu;
    Lf = log(ppft + 1e-8) + k - 0.5*log( hft + 1e-8 ) - 0.5*(nu+1) * log( 1 + (uft .* uft) ./ (hft * (nu-2) + 1e-8) );
    cf = max(Lf);
    
    Lf = exp(Lf - cf);
    lpdf_fore[1] = cf + log(Lf[1,1] + Lf[2,1]); //log predictive density (log-sum-exp)
    
  
    //t > 1
    for (t in 2:n_steps_ahead) {
      hft[1,1] = omega1 + alpha1 * uft[1,1] * uft[1,1] + beta1 * hft[1,1]; //reg 1
      hft[2,1] = omega2 + alpha2 * uft[2,1] * uft[2,1] + beta2 * hft[2,1]; //reg 2
      
      ppft = P*ppft;
      
      yft[1,1] = student_t_rng(nu, mu, sqrt(hft[1,1] + 1e-8) );
      yft[2,1] = student_t_rng(nu, mu, sqrt(hft[2,1] + 1e-8) ); 
      
      y_fore[t] = yft[1,1]*ppft[1,1] + yft[2,1]*ppft[2,1];  //collapse regs
      h_fore[t] = hft[1,1]*ppft[1,1] + hft[2,1]*ppft[2,1];  //collapse regs
      
      uft = yft - mu;
      Lf = log(ppft + 1e-8) + k - 0.5*log( hft + 1e-8 ) - 0.5*(nu+1) * log( 1 + (uft .* uft) ./ (hft * (nu-2) + 1e-8) );
      cf = max(Lf);
      
      Lf = exp(Lf - cf);
      lpdf_fore[t] = cf + log(Lf[1,1] + Lf[2,1]); //log predictive density (log-sum-exp)
    }
}


