# -------------------------------------------------------------
# Toy‑Data Factory
# -------------------------------------------------------------
library(reticulate)   # para salvar .npy
library(MASS)         # rnegbin
library(RcppCNPy)     # alternativa p/ npy (backup)

set.seed(1056)        # reprodutibilidade
n <- 100              # obs por conjunto
dir.create("toydata", showWarnings = FALSE)   # pasta‑alvo

np <- import("numpy") # interface ao NumPy


x  <- runif(n, -4, 10)
y  <- 3 + 2 * x + rnorm(n, 0, 1)
df <- data.frame(x, y)
save_all(df, "Case1")

x <- runif(nobs, 1, 6)
transition <- 0.5 * (1 + tanh(5 * (x - 3)))
mu <- (1 - 4 * x - 0.5 * x^2) * (1 - transition) +
  (-7 * sin(x^2) * 5 * log(x) + 12 * cos(2 * x)) * transition
y <- rnorm(nobs, mu, sd = 5)
df <- data.frame(x, y)
save_all(df, "Case2")



x <- runif(n, 0, 2)
mu <- 2 - 3 * x - 3 *sin(x^2) + 4*log(2*x)
p <- 1 / (1 + exp(-mu))
y <-  rbinom(n, size = 1, prob = p)
df <- data.frame(x, y)
save_all(df, "Case3")






# 1. Simulate two rings for a strongly nonlinear boundary
n_per <- 500
# angles uniform
theta <- runif(2*n_per, 0, 2*pi)
# radii: half in inner circle (r<3), half in outer annulus (4<r<6)
r_inner <- sqrt(runif(n_per, 0, 3^2))
r_outer <- sqrt(runif(n_per, 4^2, 6^2))
r <- c(r_inner, r_outer)
x1 <- r * cos(theta)
x2 <- r * sin(theta)
y  <- factor(rep(c(0,1), each = n_per))

df <- data.frame(x1 = x1, x2 = x2, y = y)
write.csv(df, "Case4", row.names = FALSE)
np$save("Case4", as.matrix(df))  #





xx <- seq(0, 2, length = 100)
mu_true <- 2 - 3 * xx - 3 *sin(xx^2) + 4*log(2*xx)
p_true <- 1 / (1 + exp(-mu_true))

fitRF <- randomForest(as.factor(y) ~ x, ntree = 5000)
ypredRF <- predict(fitRF, newdata = list(x = xx), type = "prob")[,2]

# Visualização
plot(x, y, pch = 19, col = "red", main = "Random Forest")
lines(xx, ypredRF, col = 'black', lwd = 4, lty = 2)
lines(xx, p_true, col = 'blue', lwd = 2)

library(mgcv)
fit_gam <- gam(y ~ s(x), family = binomial)
plot(x, y, pch = 19, col = "red", main = "GAM")
pred_gam <- predict(fit_gam, newdata = data.frame(x = xx), type = "response")
lines(xx, pred_gam, col = 'darkgreen', lwd = 2, lty = 3)
lines(xx, p_true, col = 'blue', lwd = 2)


