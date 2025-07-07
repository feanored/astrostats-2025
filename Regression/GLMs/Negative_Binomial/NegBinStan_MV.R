library(rstan)
library(tidyverse)
library(ggthemes)
library(viridis)

# Read Data
df <- read.csv("https://raw.githubusercontent.com/COINtoolbox/NB_GCs/refs/heads/master/Dataset/GCs_full.csv",header = TRUE, dec = ".", sep = "")
df <- subset(df, !is.na(MV_T))

#--- Ensure measurement errors are strictly positive ---------
MV_err_safe  <- pmax(df$err_MV_T, 1e-3)   # floor at 1e-3 mag
NGC_err_safe <- pmax(df$N_GC_err,  1e-3)  # not used in the likelihood yet

M   <- 2000
MVx <- seq(-24, -13, length.out = M)

stan_data <- list(
  N          = nrow(df),
  MV         = df$MV_T,
  MV_err     = MV_err_safe,
  N_GC       = df$N_GC,
  MVx        = MVx,
  M          = M,
  # dispersion prior hyper‑parameters (for a weakly‑informative gamma)
  a_size     = 0.01,
  b_size     = 0.01
)

# ------------------------------------------------------------
# Stan model as a string (clamped eta & NB2)
# ------------------------------------------------------------

stan_code <- "data {
  int<lower=1> N;                  // # galaxies
  vector[N] MV;                    // observed M_V
  vector<lower=0>[N] MV_err;       // photometric errors (mag)
  int<lower=0> N_GC[N];            // observed globular cluster counts

  int<lower=1> M;                  // # prediction points
  vector[M] MVx;                   // grid of M_V for prediction

  real<lower=0> a_size;            // NB dispersion prior hyper‑params
  real<lower=0> b_size;
}

parameters {
  real beta0;                                   // intercept
  real beta1;                                   // slope
  real<lower=0.001> size;                       // dispersion (NB2)
  vector<lower=-25, upper=-11>[N] MVtrue;        // latent true M_V
}

transformed parameters {
  vector[N] eta_raw = beta0 + beta1 * MVtrue;    // unconstrained predictor
  vector[N] eta     = fmin(20, fmax(-20, eta_raw)); // clamp to avoid overflow
  vector[N] mu      = exp(eta);                  // mean of NB2 (always > 0)
}

model {
  // --- Priors ------------------------------------------------
  beta0 ~ normal(0, 10);
  beta1 ~ normal(0, 10);
  size  ~ gamma(a_size, b_size);                // weakly‑informative

  // --- Likelihood -------------------------------------------
  MV ~ normal(MVtrue, MV_err);
  for (i in 1:N)
    N_GC[i] ~ neg_binomial_2(mu[i], size);
}

generated quantities {
  vector[M]   mu_pred;
  int         N_pred[M];

  for (j in 1:M) {
    real eta_pred  = beta0 + beta1 * MVx[j];
    real eta_c     = fmin(20, fmax(-20, eta_pred));
    mu_pred[j]     = exp(eta_c);
    N_pred[j]      = neg_binomial_2_rng(mu_pred[j], size);
  }
}"

# ------------------------------------------------------------
# Compile & sample
# ------------------------------------------------------------
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

fit <- stan(
  model_code = stan_code,
  data       = stan_data,
  chains     = 4,
  iter       = 4000,
  warmup     = 2000,
  seed       = 1234,
  control    = list(adapt_delta = 0.95, max_treedepth = 12)
)

print(fit, pars = c("beta0", "beta1", "size"), probs = c(0.025, 0.5, 0.975))

# ------------------------------------------------------------
# Posterior summaries for mu_pred
# ------------------------------------------------------------
posterior <- extract(fit)

mu_df <- tibble(
  MVx   = MVx,
  mean  = apply(posterior$mu_pred, 2, median),
  lwr50 = apply(posterior$mu_pred, 2, quantile, probs = 0.25),
  upr50 = apply(posterior$mu_pred, 2, quantile, probs = 0.75),
  lwr95 = apply(posterior$mu_pred, 2, quantile, probs = 0.025),
  upr95 = apply(posterior$mu_pred, 2, quantile, probs = 0.975)
)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
ggplot() +
  geom_ribbon(data = mu_df, aes(x = MVx, y = mean, ymin = lwr95, ymax = upr95), fill = "grey90") +
  geom_ribbon(data = mu_df, aes(x = MVx, y = mean, ymin = lwr50, ymax = upr50), fill = "grey60") +
  geom_step(data = mu_df, aes(x = MVx, y = mean), color = "black", linetype = "dashed", size = 1.2) +
  geom_point(data = df, aes(x = MV_T, y = N_GC, fill = Type, shape = Type), shape = 21, size = 3, alpha = 0.85) +
  geom_errorbar(data = df, aes(x = MV_T, y = N_GC, ymin = pmax(N_GC - NGC_err_safe, 0), ymax = N_GC + NGC_err_safe, colour = Type), width = 0.05) +
  geom_errorbarh(data = df, aes(y = N_GC, xmin = MV_T - MV_err_safe, xmax = MV_T + MV_err_safe, colour = Type), height = 0.05) +
  scale_y_continuous(trans = "asinh", breaks = c(0, 10, 100, 1e3, 1e4, 1e5),
                     labels = c("0", expression(10^1), expression(10^2), expression(10^3), expression(10^4), expression(10^5))) +
  scale_x_reverse() +
  labs(x = expression(log~M[V]), y = expression(N[GC])) +
  theme_bw() +
  scale_fill_viridis_d(name = "") +
  scale_color_viridis_d(name = "") +
  theme(text = element_text(size = 22), legend.position = "top")
