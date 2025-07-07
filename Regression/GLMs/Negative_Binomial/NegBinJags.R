# Bayesian NB regression using JAGS — Predicting Mean Relation
# Rafael S. de Souza, Bart Buelens, Ewan Cameron, Joseph Hilbe

# ------------------------------------------------------------
# Required Libraries
# ------------------------------------------------------------
require(R2jags)
library(ggplot2)
library(ggthemes)
library(Cairo)
library(ggmcmc)
library(plyr)

# ------------------------------------------------------------
# Read Data
# ------------------------------------------------------------
GCS <- read.csv("https://raw.githubusercontent.com/COINtoolbox/NB_GCs/refs/heads/master/Dataset/GCs.csv",header = TRUE, dec = ".", sep = "")
GCS <- subset(GCS, !is.na(Mdyn))
MBHx <- seq(6, 11, length.out = 1000)

## ---------- in R (before calling JAGS) ----------
jags.data <- list(
  N = nrow(GCS),
  MBH = GCS$MBH,
  errMBH = GCS$upMBH,
  errN_GC = GCS$N_GC_err,
  MBHx = MBHx,
  N_GC = GCS$N_GC,
  M = 1000
)

## ---- 1.  Build a consistent initial list -------------------
init_fun <- function() {
   list(beta0  = 1,
       beta1  = 1,
       size   = 1)          # <- always within (L , U]
}

## ---- 2.  keep the model unchanged --------------------------
model.NB <- "
model{
  beta0 ~ dnorm(0,1.0E-6)
  beta1 ~ dnorm(0,1.0E-6)
  size  ~ dunif(0.001,10)

  meanx ~ dgamma(0.01,0.01)
  varx  ~ dgamma(0.01,0.01)

  for(i in 1:N){
    MBHtrue[i] ~ dgamma(meanx^2/varx, meanx/varx)
    MBH[i]     ~ dnorm(MBHtrue[i], 1/pow(errMBH[i],2))

    eta[i] <- beta0 + beta1*MBHtrue[i]
    mu[i]  <- exp(eta[i])
    p[i]   <- size/(size+mu[i])

    N_true[i] ~ dnegbin(p[i], size)
    ## likelihood for the noisy observation
    N_GC[i] ~ dnorm(N_true[i], 1 / pow(errN_GC[i], 2))
  }

  for(j in 1:M){
    mu_pred[j] <- exp(beta0 + beta1*MBHx[j])
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
  n.burnin  = 5000,
  n.iter  = 30000,
  parameters = c("beta0","beta1","size","mu_pred","N_pred")
)




mux <- jagsresults(fit, params=c('N_pred'))

## build plotting frame (adjust the row slice if needed)
mu_df <- tibble(
  MBHx   = MBHx,                                # predictor grid
  mean   = mux[,"50%"],     # posterior mean
  lwr50  = mux[,"25%"],
  upr50  = mux[,"75%"],
  lwr95  = mux[,"2.5%"],
  upr95  = mux[,"97.5%"]
)




# ------------------------------------------------------------
# Plot Mean Relation
# ------------------------------------------------------------
#CairoPDF("MBHx_mean_relation.pdf", height = 8, width = 9)
ggplot() +
  geom_ribbon(data=mu_df,aes(x = MBHx, y = mean,ymin = lwr95, ymax = upr95), fill = "grey90") +
  geom_ribbon(data=mu_df,aes(x = MBHx, y = mean,ymin = lwr50, ymax = upr50), fill = "grey60") +
 # geom_line(data=mu_df,aes(x = MBHx, y = mean), color = "black", linetype = "dashed", size = 1.2) +
  geom_step(data=mu_df, aes(x = MBHx, y = mean), color = "black", linetype = "dashed", size = 1.2)+

  geom_point(data = GCS, aes(x = MBH, y = N_GC, fill = Type, shape = Type), shape=21,size = 3, alpha = 0.85) +
  geom_errorbar(data = GCS, aes(x = MBH, y = N_GC, ymin = pmax(N_GC - N_GC_err, 0), ymax = N_GC + N_GC_err, colour = Type), width = 0.05) +
  geom_errorbarh(data = GCS, aes(y = N_GC, xmin = MBH - lowMBH, xmax = MBH + upMBH, colour = Type), height = 0.05) +
  scale_y_continuous(trans = "asinh",
                     breaks = c(0, 10, 100, 1e3, 1e4, 1e5),
                     labels = c("0", expression(10^1), expression(10^2),
                                expression(10^3), expression(10^4), expression(10^5))) +
  labs(x = expression(log~M[BH]/M['\u0298']), y = expression(N[GC])) +
  theme_bw() +
  scale_fill_viridis_d(name="")+
  scale_color_viridis_d(name="")+
  theme(text = element_text(size = 22), legend.position = "top")
#dev.off()



# summarise *integer* quantiles
pi_df <- as_tibble(mux) |>
  mutate(MBHx = MBHx) |>
  transmute(
    MBHx,
    q05 = round(`2.5%`),
    q25 = round(`25%`),
    q50 = round(`50%`),
    q75 = round(`75%`),
    q95 = round(`97.5%`)
  )

ggplot() +
  geom_linerange(data = pi_df,
                 aes(x = MBHx, ymin = q05, ymax = q95),
                 colour = "grey70", size = 0.8) +          # 95 %
  geom_linerange(data = pi_df,
                 aes(x = MBHx, ymin = q25, ymax = q75),
                 colour = "grey30", size = 1.2) +          # 50 %
  geom_step(data = pi_df, aes(x = MBHx, y = q50),
            linetype = "dashed") +
  geom_point(data = GCS,
             aes(x = MBH, y = N_GC, fill = Type, shape = Type),
             size = 3, alpha = .85) +
  geom_errorbar(data = GCS, aes(x = MBH, y = N_GC, ymin = pmax(N_GC - N_GC_err, 0), ymax = N_GC + N_GC_err, colour = Type), width = 0.05) +
  geom_errorbarh(data = GCS, aes(y = N_GC, xmin = MBH - lowMBH, xmax = MBH + upMBH, colour = Type), height = 0.05) +
  scale_y_continuous(trans = "asinh", breaks = c(0,10,100,1e3,1e4,1e5),
                     labels = c("0", expression(10^1), expression(10^2),
                                expression(10^3), expression(10^4), expression(10^5))) +
  scale_fill_viridis_d(name="") + scale_color_viridis_d(name="") +
  labs(x = expression(log~M[BH]/M['\u0298']), y = expression(N[GC])) +
  theme_bw(base_size = 22) + theme(legend.position = "top")


draw_df  <- as_tibble(mux) |>
  mutate(MBHx = MBHx) |>
  pivot_longer(cols = starts_with("V"),        # columns holding draws
               names_to = "draw", values_to = "y") |>
  mutate(y = as.integer(round(y)))

pmf_df <- count(draw_df, MBHx, y) |>
  group_by(MBHx) |>
  mutate(prob = n / sum(n))

ggplot(pmf_df, aes(x = MBHx, y = y, fill = prob)) +
  geom_tile(height = 1, width = (MBHx[2] - MBHx[1])) +
  scale_fill_viridis_c(option = "C", name = "Pr") +
  geom_point(data = GCS,
             aes(x = MBH, y = N_GC, shape = Type),
             size = 2.5, colour = "white", stroke = .3) +
  labs(x = expression(log~M[BH]/M['\u0298']),
       y = expression(N[GC]),
       title = "Posterior predictive mass function") +
  theme_bw(base_size = 22) + theme(legend.position = "right")
