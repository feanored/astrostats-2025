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

save_all <- function(df, name) {
  csv  <- file.path("toydata", paste0(name, ".csv"))
  npy  <- file.path("toydata", paste0(name, ".npy"))
  write.csv(df, csv, row.names = FALSE)
  np$save(npy, as.matrix(df))  # matriz simples p/ Python
}

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


