# -------------------------------------------------------------
# Simulação de Dados Binários e Comparação: Logística vs RF vs MLP
# -------------------------------------------------------------
set.seed(1056)
nobs <- 400

# -------------------------------------------------------------
# 1. Geração dos Dados Binários
# -------------------------------------------------------------
x1 <- runif(nobs, 0, 6)                     # Covariável uniforme
mu <- 1 - 3 * x1 + 8 * sin(x1^2)            # Função determinística
p <- 1 / (1 + exp(-mu))                    # Transformação logística
y <- rbinom(nobs, size = 1, prob = p)       # Resposta binária

# Visualização dos dados
plot(x1, y, pch = 19, col = "red", main = "Dados binários (com jitter)")

# Grid para predições
xx <- seq(0, 6, length = 200)
mu_true <- 1 - 3 * xx + 8 * sin(xx^2)
p_true <- 1 / (1 + exp(-mu_true))

# -------------------------------------------------------------
# 2. Ajuste com Regressão Logística
# -------------------------------------------------------------
fit_logit <- glm(y ~ x1 + I(x1^3) + I(sin(x1^2)), family = binomial)
summary(fit_logit)

ypred_logit <- predict(fit_logit, newdata = list(x1 = xx), type = "response")

# Visualização
plot(x1, y, pch = 19, col = "red", main = "Logística")
lines(xx, ypred_logit, col = 'black', lwd = 4, lty = 2)
lines(xx, p_true, col = 'blue', lwd = 2)     # Probabilidade verdadeira

# -------------------------------------------------------------
# 3. Ajuste com Random Forest
# -------------------------------------------------------------
library(randomForest)
fitRF <- randomForest(as.factor(y) ~ x1, ntree = 5000)
ypredRF <- predict(fitRF, newdata = list(x1 = xx), type = "prob")[,2]

# Visualização
plot(x1, y, pch = 19, col = "red", main = "Random Forest")
lines(xx, ypredRF, col = 'black', lwd = 4, lty = 2)
lines(xx, p_true, col = 'blue', lwd = 2)

# -------------------------------------------------------------
# 4. Ajuste com Rede Neural (MLP)
# -------------------------------------------------------------
library(nnet)

x1_df <- data.frame(x1 = x1)
xx_df <- data.frame(x1 = xx)

nn_mod <- nnet::nnet(
  x = x1_df,
  y = class.ind(y),          # Codifica y como 0/1 (necessário p/ classificação)
  size = 100,
  softmax = TRUE,            # Para classificação
  maxit = 1000,
  decay = 1e-4,
  trace = FALSE
)

ypredMLP <- predict(nn_mod, newdata = xx_df)[,2]  # Probabilidade da classe 1

# Visualização
plot(x1, y, pch = 19, col = "red", main = "Rede Neural")
lines(xx, ypredMLP, col = 'black', lwd = 4, lty = 2)
lines(xx, p_true, col = 'blue', lwd = 2)




library(mgcv)
fit_gam <- gam(y ~ s(x1), family = binomial)
plot(x1, y, pch = 19, col = "red", main = "GAM")
pred_gam <- predict(fit_gam, newdata = data.frame(x1 = xx), type = "response")
lines(xx, pred_gam, col = 'darkgreen', lwd = 2, lty = 3)
lines(xx, p_true, col = 'blue', lwd = 2)




semean <- function(x) Hmisc::binconf(sum(x == 1), length(x))

bin_dat <- function(x, y, binx = 0.25){
  t.breaks <- cut(x, seq(min(x), max(x), by = binx))
  means <- tapply(y, t.breaks, mean)
  means.se <- tapply(y, t.breaks, semean) %>% do.call(rbind.data.frame, .)
  gbin <- data.frame(
    x = seq(binx + min(x), max(x), by = binx),
    y = means,
    ymin = means.se$Lower,
    ymax = means.se$Upper
  )
  return(gbin)
}

gbin <- bin_dat(x1, y)

# ---------------------------
# Gráfico com ggplot
# ---------------------------

ggplot() +
  geom_point(aes(x = x1, y = y), color = "red", shape = 16, position = position_jitter(height = 0.02)) +
#  geom_line(aes(x = xx, y = pred_gam), color = "darkgreen", lwd = 1.5, linetype = "dashed") +
#  geom_line(aes(x = xx, y = p_true), color = "blue", lwd = 1) +
#  geom_errorbar(data = gbin, aes(x = x, y = y, ymin = ymin, ymax = ymax), width = 0.05, color = "#9C4045") +
#  geom_point(data = gbin, aes(x = x, y = y), size = 3, fill = "#9C4045", shape = 21, color = "orange") +
#  labs(title = "GAM fit with binned data and confidence intervals",
#       x = "x1", y = "Probability") +
  theme_xkcd()



ggplot() +
  geom_point(aes(x = x1, y = y), color = "red", shape = 16, position = position_jitter(height = 0.03)) +
  geom_line(aes(x = xx, y = pred_gam), color = "darkgreen", lwd = 1.5, linetype = "dashed") +
  geom_line(aes(x = xx, y = p_true), color = "blue", lwd = 1) +
  geom_errorbar(data = gbin, aes(x = x, y = y, ymin = ymin, ymax = ymax), width = 0.05, color = "#9C4045") +
  geom_point(data = gbin, aes(x = x, y = y), size = 3, fill = "#9C4045", shape = 21, color = "orange") +
  labs(x = "x1", y = "Probability") +
  theme_xkcd()
