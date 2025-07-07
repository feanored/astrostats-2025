# Bayesian NB regression using JAGS — Predicting Mean Relation
# Rafael S. de Souza, Bart Buelens, Ewan Cameron, Joseph Hilbe

# ------------------------------------------------------------
# Required Libraries
# ------------------------------------------------------------
library(rjags)
library(runjags)
library(ggplot2)
library(ggthemes)
library(scales)
library(Cairo)
library(ggmcmc)
library(plyr)
library(gridExtra)
library(pander)
library(parallel)

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
asinh_trans <- function() trans_new('asinh', asinh, sinh)

facet_wrap_labeller <- function(gg.plot, labels=NULL) {
  g <- ggplotGrob(gg.plot)
  gg <- g$grobs
  strips <- grep("strip_t", names(gg))
  for(ii in seq_along(labels)) {
    modgrob <- getGrob(gg[[strips[ii]]], "strip.text", grep=TRUE, global=TRUE)
    gg[[strips[ii]]]$children[[modgrob$name]] <- editGrob(modgrob,label=labels[ii])
  }
  g$grobs <- gg
  class(g) = c("arrange", "ggplot",class(g))
  g
}

# ------------------------------------------------------------
# Read Data
# ------------------------------------------------------------
GCS <- read.csv("https://raw.githubusercontent.com/COINtoolbox/NB_GCs/refs/heads/master/Dataset/GCs.csv",header = TRUE, dec = ".", sep = "")
GCS <- subset(GCS, !is.na(Mdyn))

N <- nrow(GCS)
MBHx <- seq(0.95 * min(GCS$MBH), 1.05 * max(GCS$MBH), length.out = 500)

jags.data <- list(
  N = N,
  N_GC = GCS$N_GC,
  MBH = GCS$MBH,
  errN_GC = GCS$N_GC_err,
  errMBH = GCS$upMBH,
  MBHx = MBHx,
  M = 500
)

# ------------------------------------------------------------
# JAGS Model (Mean Prediction)
# ------------------------------------------------------------
model.NB <- "model {
  beta.0 ~ dnorm(0, 1.0E-6)
  beta.1 ~ dnorm(0, 1.0E-6)
  size   ~ dunif(0.001, 10)

  meanx ~ dgamma(0.01, 0.01)
  varx  ~ dgamma(0.01, 0.01)

  for (i in 1:N) {
    MBHtrue[i] ~ dgamma(meanx^2 / varx, meanx / varx)
    MBH[i]     ~ dnorm(MBHtrue[i], 1 / pow(errMBH[i], 2))

    eta[i] <- beta.0 + beta.1 * MBHtrue[i]
    mu[i]  <- exp(eta[i]) + errorN[i] - errN_GC[i]
    p[i]   <- size / (size + mu[i])

    N_GC[i] ~ dnegbin(p[i], size)
    errorN[i] ~ dbin(0.5, 2 * errN_GC[i])
  }

  for (j in 1:M) {
    etax[j] <- beta.0 + beta.1 * MBHx[j]
    mux[j]  <- exp(max(-20, min(20, etax[j])))
    mu_pred[j] <- mux[j]
  }
}
"

inits <- function() list(beta.0 = rnorm(1), beta.1 = rnorm(1), size = runif(1, 0.1, 5))

# ------------------------------------------------------------
# Run Model
# ------------------------------------------------------------
jags.fit <- run.jags(model = model.NB,
                     data = jags.data,
                     inits = list(inits(), inits(), inits()),
                     n.chains = 3,
                     adapt = 2000,
                     burnin = 20000,
                     sample = 50000,
                     monitor = c("beta.0", "beta.1", "size", "mu_pred"),
                     summarise = FALSE,
                     plots = FALSE)

# ------------------------------------------------------------
# Extract Posterior Summaries for Mean Prediction
# ------------------------------------------------------------
library(coda)
library(dplyr)


# Extract summary
summ <- summary(as.mcmc.list(jags.fit), var = "mu_pred", quantiles = c(0.025, 0.5, 0.975))

# Create dataframe for plotting the mean relation
mu_df <- tibble(
  MBHx = MBHx,
  mean = summ$statistics[4:503, "Mean"],
  lwr  = summ$quantiles[4:503, 1],  # 2.5%
  upr  = summ$quantiles[4:503, 3]   # 97.5%
)



# ------------------------------------------------------------
# Plot Mean Relation
# ------------------------------------------------------------
CairoPDF("MBHx_mean_relation.pdf", height = 8, width = 9)
ggplot() +
  geom_ribbon(data=mu_df,aes(x = MBHx, y = mean,ymin = lwr, ymax = upr), fill = "grey70", alpha = 0.3) +
  geom_line(data=mu_df,aes(x = MBHx, y = mean), color = "black", linetype = "dashed", size = 1.2) +
  geom_point(data = GCS, aes(x = MBH, y = N_GC, colour = Type, shape = Type), size = 3, alpha = 0.85) +
  geom_errorbar(data = GCS, aes(x = MBH, y = N_GC, ymin = pmax(N_GC - N_GC_err, 0), ymax = N_GC + N_GC_err, colour = Type), width = 0.05) +
  geom_errorbarh(data = GCS, aes(y = N_GC, xmin = MBH - lowMBH, xmax = MBH + upMBH, colour = Type), height = 0.05) +
  scale_y_continuous(trans = "asinh",
                     breaks = c(0, 10, 100, 1000, 10000),
                     labels = c("0", expression(10^1), expression(10^2), expression(10^3), expression(10^4))) +
  labs(x = expression(log~M[BH]/M['\u0298']), y = expression(N[GC])) +
  theme_hc() +
  theme(text = element_text(size = 22), legend.position = "top")
dev.off()
