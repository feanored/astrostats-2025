############################################################
# Bayesian NB regression with measurement error in NIMBLE
# (adapted from the JAGS code by R. S. de Souza et al.)
############################################################

# ------------------------------------------------------------
# 1.  Libraries
# ------------------------------------------------------------
library(nimble)          # MCMC engine
library(coda)            # convenient summary
library(tidyverse)       # tibble, ggplot2, dplyr, etc.
library(Cairo)           # optional PDF backend

# ------------------------------------------------------------
# 2.  Read & pre-process data
# ------------------------------------------------------------
# Read Data
df <- read.csv("https://raw.githubusercontent.com/COINtoolbox/NB_GCs/refs/heads/master/Dataset/GCs_full.csv",header = TRUE, dec = ".", sep = "")
df <- subset(df, !is.na(MV_T))


M      <- 2000
MVx    <- seq(-25, -10, length.out = M)

# observation uncertainties: enforce >0
df$N_GC_err <- pmax(df$N_GC_err, 0.5)

# ------------------------------------------------------------
# 3.  Constants & data for NIMBLE
# ------------------------------------------------------------
constants <- list(
  N  = nrow(df),
  M  = M
)

dataList <- list(
  MV           = df$MV_T,
  MV_err   =    df$err_MV_T,            # precision for dnorm
  N_GC         = df$N_GC,
  N_GC_err = df$N_GC_err,            # Gaussian obs error
  MVx          = MVx
)

# ------------------------------------------------------------
# 4.  NIMBLE model code
# ------------------------------------------------------------
code <- nimbleCode({
  # Priors
  beta0 ~ dnorm(0, var = 1e2)                  # sd ≈ 10
  beta1 ~ dnorm(0, var = 1e2)
  size  ~ dunif(0.001, 10)
  
  # Latent–truth loop
  for(i in 1:N) {
    MVtrue[i] ~ dunif(-25, -11)
    MV[i]     ~ dnorm(MVtrue[i], sd = MV_err[i])
    
    eta[i] <- beta0 + beta1 * MVtrue[i]
    mu[i]  <- exp(eta[i])
    p[i]   <- size / (size + mu[i])
    
    N_true[i] ~ dnegbin(p[i], size = size)
    
    # Gaussian measurement error on observed counts
    N_GC[i] ~ dnorm(N_true[i], sd = N_GC_err[i])
  }
  
  # Prediction grid
  for(j in 1:M) {
    eta_pred[j] <- beta0 + beta1 * MVx[j]
    mu_pred[j]  <- exp(max(-20, min(20, eta_pred[j])))   # clip
    p_pred[j]   <- size / (size + mu_pred[j])
    N_pred[j]   ~ dnegbin(p_pred[j], size = size)
  }
})

# ------------------------------------------------------------
# 5.  Initial values
# ------------------------------------------------------------
initFun <- function() list(
  beta0   = rnorm(1),
  beta1   = rnorm(1),
  size    = runif(1, 0.5, 2),
  MVtrue  = rnorm(constants$N, df$MV_T, df$err_MV_T),
  N_true  = pmax(1, df$N_GC),          # avoid zero counts initially
  N_pred  = rep(1, constants$M)
)

# ------------------------------------------------------------
# 6.  Build, compile, and run MCMC
# ------------------------------------------------------------
model      <- nimbleModel(code, data = dataList,
                          constants = constants,
                          inits = initFun())
cModel     <- compileNimble(model)

mcmcConf   <- configureMCMC(model, monitors = c(
  "beta0", "beta1", "size", "mu_pred", "N_pred"))
mcmc       <- buildMCMC(mcmcConf)
cMCMC      <- compileNimble(mcmc, project = model)

set.seed(123)
samples <- runMCMC(cMCMC,
                   niter   = 50000,
                   nburnin = 20000,
                   nchains = 3,
                   samplesAsCodaMCMC = TRUE,
                   summary = FALSE,
                   WAIC    = FALSE)

# ------------------------------------------------------------
# 7.  Posterior summaries
# ------------------------------------------------------------
# Combine chains & summarise mu_pred
mu_idx <- grep("^mu_pred\\[", varnames(samples))
mu_mat <- do.call(rbind, samples)[, mu_idx]

summ_mu <- apply(mu_mat, 2, quantile,
                 probs = c(.025, .25, .5, .75, .975)) %>% t() %>% as_tibble()
names(summ_mu) <- c("lwr95","lwr50","mean","upr50","upr95")
summ_mu$MVx <- MVx

# Combine chains & summarise N_pred (for discrete PI plot)
N_idx  <- grep("^N_pred\\[", varnames(samples))
N_mat  <- do.call(rbind, samples)[, N_idx]

summ_N <- apply(N_mat, 2, quantile,
                probs = c(.025, .25, .5, .75, .975)) %>% t() %>% as_tibble()
names(summ_N) <- c("lwr95","lwr50","mean","upr50","upr95")
summ_N$MVx <- MVx

# ------------------------------------------------------------
# 8.  Plot mean–relation with mu_pred (continuous expectation)
# ------------------------------------------------------------
plog <- scales::pseudo_log_trans(base = 10, sigma = 0.5)

p1 <- ggplot() +
  geom_point(data = df,
             aes(x = MV_T, y = N_GC, fill = Type,shape=Type,color=Type),
             size = 3, alpha = 0.85) +
  geom_errorbar(data = df,
                aes(x = MV_T, ymin = pmax(N_GC - N_GC_err, 0),
                    ymax = N_GC + N_GC_err, colour = Type),
                width = 0.1) +
  geom_errorbarh(data = df,
                 aes(y = N_GC, xmin = MV_T - err_MV_T,
                     xmax = MV_T + err_MV_T, colour = Type),
                 height = 0.1) +
  geom_ribbon(data = summ_mu,
              aes(x = MVx, ymin = lwr95, ymax = upr95),
              fill = "grey90") +
  geom_ribbon(data = summ_mu,
              aes(x = MVx, ymin = lwr50, ymax = upr50),
              fill = "grey60") +
  geom_step(data = summ_mu,
            aes(x = MVx, y = mean),
            linetype = "dashed", size = 1) +
  scale_y_continuous(trans = plog,
                     breaks = c(0, 10, 100, 1e3, 1e4, 1e5),
                     labels = c("0", expression(10^1), expression(10^2),
                                expression(10^3), expression(10^4), expression(10^5))) +
  scale_x_reverse() +
  labs(x = expression(log~M[V]),
       y = expression(N[GC])) +
  scale_fill_viridis_d(name="") +
  scale_color_viridis_d(name="") +
  scale_shape_discrete(name="") +
  theme_bw(base_size = 22) +
  theme(legend.position = "top")
print(p1)

# ------------------------------------------------------------
# 9.  Plot discrete prediction intervals with N_pred
# ------------------------------------------------------------
p2 <- ggplot() +
  geom_linerange(data = summ_N,
                 aes(x = MVx, ymin = lwr95, ymax = upr95),
                 colour = "grey70") +
  geom_linerange(data = summ_N,
                 aes(x = MVx, ymin = lwr50, ymax = upr50),
                 colour = "grey30", size = 1.2) +
  geom_step(data = summ_N,
            aes(x = MVx, y = mean), linetype = "dashed") +
  geom_point(data = df,
             aes(x = MV_T, y = N_GC, fill = Type, shape = Type),
             size = 3, alpha = 0.85) +
  scale_y_continuous(trans = plog,
                     breaks = c(0, 10, 100, 1e3, 1e4, 1e5),
                     labels = c("0", expression(10^1), expression(10^2),
                                expression(10^3), expression(10^4), expression(10^5))) +
  scale_x_reverse()+
  labs(x = expression(log~M[V]), y = expression(N[GC])) +
  scale_fill_viridis_d(name="") + scale_color_viridis_d(name="") +
  theme_bw(base_size = 22) + theme(legend.position = "top")
print(p2)
