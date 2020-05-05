data {
  int<lower=1> N;                     // Number of subjects
  int<lower=1> T;                     // Max number of trials across subjects
  int<lower=1,upper=T> Tsubj[N];      // Number of trials/block for each subject
  int<lower=0,upper=1> decision[N, T];  // The options subjects choose (0: enter / 1: shift)
   real<lower=0> eff_e[N, T];
   real<lower=0> eff_s[N, T];
   real<lower=0> equ_e[N, T];
   real<lower=0> equ_s[N, T];
}

parameters {
  vector[3] mu_pr;
  vector<lower=0>[3] sigma;

  vector[N] alpha_pr;   // risk attitude parameter
  vector[N] lamda_pr;   // inverse temperature parameter
  vector[N] bias_pr;   
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0,upper=1>[N] alpha;
  vector<lower=0>[N] lamda;
  vector[N] bias;

  alpha = Phi_approx(mu_pr[1] + sigma[1] * alpha_pr) ;//fast approximation of the unit,p18
  lamda = exp(mu_pr[2] + sigma[2] * lamda_pr);
  bias = mu_pr[3] + sigma[3] * bias_pr;
}

model {
  // hyper parameters
  mu_pr[1]  ~ normal(0, 1.0);
  mu_pr[2]  ~ normal(0, 1.0);
  mu_pr[3]  ~ normal(0, 10.0);
  sigma~ cauchy(0, 5);

  // individual parameters 
  alpha_pr ~ normal(0, 1);
  lamda_pr ~ normal(0, 1);
  bias_pr ~ normal(0, 1);

  for (i in 1:N) {
    for (t in 1:Tsubj[i]) {
      real u_e;  // subjective value of enter
      real u_s;  // subjective value of shift
      real p_s;  // probability of choosing shift

      u_e = (1-alpha[i])*eff_e[i,t]-alpha[i]*equ_e[i,t];
      u_s = (1-alpha[i])*eff_s[i,t]-alpha[i]*equ_s[i,t];
      p_s = inv_logit(lamda[i] * (u_s - u_e + bias[i])); //logistic sigmoid function applied to x

      target += bernoulli_lpmf(decision[i, t] | p_s);//The log Bernoulli probability mass of y given chance of success theta
    }
  }
}

generated quantities {
  // For group level parameters
  real<lower=0,upper=1> mu_alpha;
  real<lower=0> mu_lamda;
  real mu_bias;

  // For log likelihood calculation for each subject
  real log_lik[N];

  // For posterior predictive check
  real y_pred[N, T];

  // Model regressors

  real<lower=0,upper=1> p_s[N, T];

  // Set all posterior predictions to -1 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  mu_alpha  = Phi_approx(mu_pr[1]) ;
  mu_lamda  = exp(mu_pr[2]);
  mu_bias  = mu_pr[3];

  { // local section, this saves time and space
    for (i in 1:N) {
      // Initialize the log likelihood variable to 0.
      log_lik[i] = 0;

      for (t in 1:Tsubj[i]) {
        real u_e;  // subjective value of enter
        real u_s;  // subjective value of shift

        u_e = (1-alpha[i])*eff_e[i,t]-alpha[i]*equ_e[i,t];
        u_s = (1-alpha[i])*eff_s[i,t]-alpha[i]*equ_s[i,t];
        p_s[i, t] = inv_logit(lamda[i] * (u_s - u_e + bias[i]));


        log_lik[i] += bernoulli_lpmf(decision[i, t] | p_s[i, t]);//The log Bernoulli probability mass of y given chance of success theta
        y_pred[i, t] = bernoulli_rng(p_s[i, t]);//Generate a Bernoulli variate with chance of success theta; may only be used in transformed data and generated quantities blocks. 
      }
    }
  }
}
