# Required Libraries
library(nimble)
library(coda)
library(ggplot2)
library(dplyr)

# Load Data
GCS <- read.csv("https://raw.githubusercontent.com/COINtoolbox/NB_GCs/refs/heads/master/Dataset/GCs.csv", header = TRUE, dec = ".", sep = "")
GCS <- subset(GCS, !is.na(Mdyn))

N <- nrow(GCS)
MBHx <- seq(0.95 * min(GCS$MBH), 1.05 * max(GCS$MBH), length.out = 500)

dataList <- list(
  N = N,
  N_GC = GCS$N_GC,
  MBH = GCS$MBH,
  errN_GC = GCS$N_GC_err,
  errMBH = GCS$upMBH,
  MBHx = MBHx,
  M = length(MBHx)
)

# Define the NIMBLE model
code <- nimbleCode({
  beta0 ~ dnorm(0, sd = 1.0E3)
  beta1 ~ dnorm(0, sd = 1.0E3)
  size  ~ dunif(0.001, 10)

  meanx ~ dgamma(0.01, 0.01)
  varx  ~ dgamma(0.01, 0.01)

  for (i in 1:N) {
    MBHtrue[i] ~ dgamma(meanx^2 / varx, meanx / varx)
    MBH[i] ~ dnorm(MBHtrue[i], sd = errMBH[i])

    eta[i] <- beta0 + beta1 * MBHtrue[i]
    log_mu[i] <- log(exp(eta[i]) + errorN[i] - errN_GC[i])
    mu[i] <- exp(log_mu[i])
    p[i] <- size / (size + mu[i])

    N_GC[i] ~ dnegbin(prob = p[i], size = size)
    errorN[i] ~ dbin(0.5, size = 2 * errN_GC[i])
  }

  for (j in 1:M) {
    eta_pred[j] <- beta0 + beta1 * MBHx[j]
    eta_capped[j] <- max(-20, min(20, eta_pred[j]))
    mu_pred[j] <- exp(eta_capped[j])
  }
})

# Initial values
inits <- function() list(beta0 = rnorm(1), beta1 = rnorm(1), size = runif(1, 0.1, 5))

constants <- list(N = dataList$N, M = dataList$M)
data <- dataList
inits_vals <- inits()

# Build and compile the model
model <- nimbleModel(code, constants = constants, data = data, inits = inits_vals)
compiled_model <- compileNimble(model)

# Configure and build MCMC
conf <- configureMCMC(model, monitors = c("beta0", "beta1", "size", "mu_pred"))
mcmc <- buildMCMC(conf)
compiled_mcmc <- compileNimble(mcmc, project = model)

# Run the model
samples <- runMCMC(compiled_mcmc, niter = 50000, nburnin = 20000, nchains = 3, samplesAsCodaMCMC = TRUE)

# Summarize posterior for mu_pred
summ <- summary(samples)$statistics
quant <- summary(samples)$quantiles

mu_df <- tibble(
  MBHx = MBHx,
  mean = summ[grep("mu_pred", rownames(summ)), "Mean"],
  lwr  = quant[grep("mu_pred", rownames(quant)), "2.5%"],
  upr  = quant[grep("mu_pred", rownames(quant)), "97.5%"]
)

# Plot
ggplot(mu_df, aes(x = MBHx, y = mean)) +
  geom_ribbon(aes(ymin = lwr, ymax = upr), fill = "grey80") +
  geom_line(color = "black") +
  theme_minimal() +
  labs(x = expression(log~M[BH]/M['ʘ']), y = expression(mu[GC]), title = "Mean Relation from NIMBLE Model")
