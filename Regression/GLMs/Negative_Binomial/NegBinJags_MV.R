# Bayesian NB regression using JAGS — Predicting Mean Relation
# Rafael S. de Souza, Bart Buelens, Ewan Cameron, Joseph Hilbe

# ------------------------------------------------------------
# Required Libraries
# ------------------------------------------------------------
require(R2jags)
library(rjags)
#load.module("extra")
library(ggplot2)
library(ggthemes)
library(Cairo)
library(ggmcmc)
library(plyr)
source("jagsresults.R")
plog <- scales::pseudo_log_trans(base = 10, sigma = 5)

# ------------------------------------------------------------
# Read Data
# ------------------------------------------------------------
df <- read.csv("https://raw.githubusercontent.com/COINtoolbox/NB_GCs/refs/heads/master/Dataset/GCs_full.csv",header = TRUE, dec = ".", sep = "")
df <- subset(df, !is.na(MV_T))
M <- 2000
MVx <- seq(-25,-10,length.out = M)

## ---------- in R (before calling JAGS) ----------
jags.data <- list(
  N = nrow(df),
  MV = df$MV_T,
  MV_err = df$err_MV_T,
  N_GC_err = pmax(df$N_GC_err, 0.5),
  MVx = MVx,
  N_GC = df$N_GC,
  M = M
)

## ---- 1.  Build a consistent initial list -------------------
init_fun <- function() {
   list(beta0  = 1,
       beta1  = 1,
       size   = 1,
       N_pred  = rep(1, M))          
}

## ---- 2.  keep the model unchanged --------------------------
model.NB <- "
model{
  beta0 ~ dnorm(0,1.0E-2)
  beta1 ~ dnorm(0,1.0E-2)
  size  ~ dunif(0.001,10)

  for(i in 1:N){
    MVtrue[i] ~ dunif(-25,-11)
    MV[i]     ~ dnorm(MVtrue[i], 1/pow(MV_err[i],2))

    eta[i] <- beta0 + beta1*MVtrue[i]
    mu[i]  <- exp(eta[i])
    p[i]   <- size/(size+mu[i])

    N_true[i] ~ dnegbin(p[i], size)
    ## likelihood for the noisy observation
    N_GC[i] ~ dnorm(N_true[i], 1/pow(N_GC_err[i], 2))
  }

  for(j in 1:M){
     eta_pred[j] <- beta0 + beta1*MVx[j]
    log(mu_pred[j]) <-  max(-20,min(20,eta_pred[j])) 
    p_pred[j]   <- size/(size+mu_pred[j])
    N_pred[j] ~ dnegbin(p_pred[j], size)
  }
}
"

## ---- 3.  Run ------------------------------------------------
fit <- jags(
  model   = textConnection(model.NB),
  data    = jags.data,
  inits   = list(init_fun(), init_fun(), init_fun()),
  n.chains= 3,
  n.burnin  = 20000,
  n.iter  = 50000,
  parameters = c("beta0","beta1","size","mu_pred","N_pred")
)


mux <- jagsresults(fit, params=c('mu_pred'))

mu_df <- tibble(
  MVx   = MVx,                                # predictor grid
  mean   = mux[,"50%"],     # posterior mean
  lwr50  = mux[,"25%"],
  upr50  = mux[,"75%"],
  lwr95  = mux[,"2.5%"],
  upr95  = mux[,"97.5%"]
)
# ------------------------------------------------------------
# Plot Mean Relation
# ------------------------------------------------------------
#CairoPDF("MV_N_GC.pdf", height = 8, width = 9)
ggplot() +
  geom_point(data = df, aes(x = MV_T, y = N_GC, fill = Type, shape = Type), shape=21,size = 3, alpha = 0.85) +
  geom_errorbar(data = df, aes(x = MV_T, y = N_GC, ymin = N_GC -max(N_GC - N_GC_err, 0), ymax = N_GC + N_GC_err, colour = Type), width = 0.05) +
  geom_errorbarh(data = df, aes(y = N_GC, xmin = MV_T - err_MV_T, xmax = MV_T + err_MV_T, colour = Type), height = 0.05) +
  geom_ribbon(data=mu_df,aes(x = MVx, y = mean,ymin = lwr95, ymax = upr95), fill = "grey90") +
  geom_ribbon(data=mu_df,aes(x = MVx, y = mean,ymin = lwr50, ymax = upr50), fill = "grey60") +
  geom_step(data=mu_df, aes(x = MVx, y = mean), color = "black", linetype = "dashed", size = 1.2)+

  
  scale_y_continuous(trans = plog,
                     breaks = c(0, 10, 100, 1e3, 1e4, 1e5),
                     labels = c("0", expression(10^1), expression(10^2),
                                expression(10^3), expression(10^4), expression(10^5))) +
  scale_x_reverse()+
  labs(x = expression(log~M[V]), y = expression(N[GC])) +
  theme_bw() +
  scale_fill_viridis_d(name="")+
  scale_color_viridis_d(name="")+
  theme(text = element_text(size = 22), legend.position = "top")
#dev.off()

N_x <- jagsresults(fit, params=c('N_pred'))

N_df <- tibble(
  MVx   = MVx,                                # predictor grid
  mean   = N_x[,"50%"],     # posterior mean
  lwr50  = N_x[,"25%"],
  upr50  = N_x[,"75%"],
  lwr95  = N_x[,"2.5%"],
  upr95  = N_x[,"97.5%"]
)
# 
ggplot() +
  geom_ribbon(data=N_df,aes(x = MVx, y = mean,ymin = lwr95, ymax = upr95), fill = "grey90") +
  geom_ribbon(data=N_df,aes(x = MVx, y = mean,ymin = lwr50, ymax = upr50), fill = "grey60") +
  geom_step(data=N_df, aes(x = MVx, y = mean), color = "black", linetype = "dashed", size = 1.2)+
  
  geom_point(data = df, aes(x = MV_T, y = N_GC, fill = Type, shape = Type), shape=21,size = 3, alpha = 0.85) +
  geom_errorbar(data = df, aes(x = MV_T, y = N_GC, ymin = pmax(N_GC - N_GC_err, 0), ymax = N_GC + N_GC_err, colour = Type), width = 0.05) +
  geom_errorbarh(data = df, aes(y = N_GC, xmin = MV_T - err_MV_T, xmax = MV_T + err_MV_T, colour = Type), height = 0.05) +
  scale_y_continuous(trans = plog,
                     breaks = c(0, 10, 100, 1e3, 1e4, 1e5),
                     labels = c("0", expression(10^1), expression(10^2),
                                expression(10^3), expression(10^4), expression(10^5))) +
  scale_x_reverse()+
  labs(x = expression(log~M[V]), y = expression(N[GC])) +
  theme_bw() +
  scale_fill_viridis_d(name="")+
  scale_color_viridis_d(name="")+
  theme(text = element_text(size = 22), legend.position = "top")
#dev.off()
