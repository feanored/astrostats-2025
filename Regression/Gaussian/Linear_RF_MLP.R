# -------------------------------------------------------------
# Simulação de Dados e Comparação: Regressão Linear vs Random Forest
# -------------------------------------------------------------
set.seed(1056)                     # Garantir reprodutibilidade
nobs <- 400                        # Número de observações

# -------------------------------------------------------------
# 1. Geração dos Dados
# -------------------------------------------------------------
x1 <- runif(nobs, 0, 6)            # Covariável uniforme entre 0 e 6
mu <- 1 - 3 * x1 + 8 * sin(x1^2)   # Linha verdadeira (modelo gerador)
y <- rnorm(nobs, mu, sd=1)         # Variável resposta com ruído normal

# -------------------------------------------------------------
# 2. Ajuste do Modelo Linear com Transformações Não-lineares
# -------------------------------------------------------------
fit <- lm(y ~ x1 + I(x1^3) + I(sin(x1^2)))  # Ajuste do modelo linear
summary(fit)                                # Resumo do ajuste

# Predição no grid
xx <- seq(0, 6, length = 200)
ypred <- predict(fit, newdata = list(x1 = xx), type = "response")

# Visualização do ajuste
plot(x1, y, pch = 19, col = "red", main = "Modelo Linear com Transformações")
lines(xx, ypred, col = 'black', lwd = 4, lty = 2)             # Linha predita pelo modelo
lines(xx, 1 - 3 * xx + 8 * sin(xx^2), col = 'blue', lwd = 2)  # Linha verdadeira
segments(x1, fitted(fit), x1, y, lwd = 2, col = "gray")       # Resíduos

# -------------------------------------------------------------
# 3. Ajuste com Random Forest
# -------------------------------------------------------------
library(randomForest)
fitRF <- randomForest(y ~ x1)       # Ajuste com Random Forest
print(fitRF)                        # Exibe resumo do modelo

# Predição no grid
ypredRF <- predict(fitRF, newdata = list(x1 = xx), type = "response")

# Visualização do ajuste
plot(x1, y, pch = 19, col = "red", main = "Random Forest")
lines(xx, ypredRF, col = 'black', lwd = 4, lty = 2)            # Linha predita pelo RF
lines(xx, 1 - 3 * xx + 8 * sin(xx^2), col = 'blue', lwd = 2)   # Linha verdadeira
#segments(x1, fitRF$predicted, x1, y, lwd = 2, col = "gray")    # Resíduos




# Supondo que x1 e y já estão definidos como no seu exemplo anterior
x1_df <- data.frame(x1 = x1)
xx_df <- data.frame(x1 = xx)

# Ajusta a rede neural com 1 entrada e 25 neurônios ocultos
nn_mod <- nnet::nnet(
  x = x1_df,
  y = y,
  size = 75,
  linout = TRUE,       # para regressão (em vez de softmax = FALSE, que é para classificação)
  maxit = 500,
  decay = 1e-4,
  trace = FALSE
)

# Predição
ypredMLP <- predict(nn_mod, newdata = xx_df)

# Visualização
plot(x1, y, pch = 19, col = "red", main = "Neural Network Fit")
lines(xx, ypredMLP, col = 'black', lwd = 4, lty = 2)              # Linha predita pela MLP
lines(xx, 1 - 3 * xx + 8 * sin(xx^2), col = 'blue', lwd = 2)     # Linha verdadeira


