# -------------------------------------------------------------
# Load Required Libraries
# -------------------------------------------------------------
library(randomForest)
library(rpart)
library(partykit)
library(ggparty)
library(ggplot2)
library(dplyr)
library(patchwork)

# -------------------------------------------------------------
# 1. Simulação dos Dados
# -------------------------------------------------------------
set.seed(1056)
nobs <- 400
x1 <- runif(nobs, 0, 6)
mu <- 1 - 3 * x1 + 8 * sin(x1^2)
y <- rnorm(nobs, mu, sd = 1)
df <- data.frame(x1 = x1, y = y)

# Grid para predições
xx <- seq(0, 6, length.out = 300)
truth <- 1 - 3 * xx + 8 * sin(xx^2)

# -------------------------------------------------------------
# 2. Experimento: Random Forest com diferentes maxnodes
# -------------------------------------------------------------
nodes_list <- c(1, 2, 5, 20, 50, 100)
maxdepth_list <- ceiling(log2(nodes_list + 1))

rf_results <- list()
rpart_models <- list()
rpart_parties <- list()
for (i in seq_along(nodes_list)) {
  n_nodes <- nodes_list[i]
  depth <- maxdepth_list[i]
  label <- paste0("maxnodes = ", n_nodes)

  # Ajuste com Random Forest
  rf_model <- randomForest(y ~ x1, data = df, ntree = 1, maxnodes = n_nodes)
  ypred <- predict(rf_model, newdata = data.frame(x1 = xx))

  rf_results[[i]] <- data.frame(
    xx = xx,
    ypred = ypred,
    model = label,
    order = n_nodes
  )

  # Ajuste com rpart (para visualização da árvore)
  rpart_fit <- rpart(y ~ x1, data = df, control = rpart.control(maxdepth = depth, cp = 0))
  rpart_models[[i]] <- rpart_fit                    # Salva o rpart puro
  rpart_parties[[i]] <- as.party(rpart_fit)         # Para o ggparty

}

# -------------------------------------------------------------
# 3. Visualização: Regressão com RF
# -------------------------------------------------------------
df_obs <- data.frame(x1 = x1, y = y)
df_pred <- do.call(rbind, rf_results)
df_pred$model <- factor(df_pred$model, levels = paste0("maxnodes = ", nodes_list))

plot_rf <- ggplot() +
  geom_point(data = df_obs, aes(x = x1, y = y), color = "gray30", alpha = 0.5) +
  geom_line(data = df_pred, aes(x = xx, y = ypred), size = 1, color = "orange2") +
  facet_wrap(~ model, ncol = 2) +
  labs(
    title = "Random Forest Regressor",
    subtitle = "Aumentando o número de nós",
    x = "x1", y = "y"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

# -------------------------------------------------------------
# 4. Visualização: Árvore de Decisão com ggparty
# -------------------------------------------------------------

par(mfrow = c(3, 2))
for (i in seq_along(rpart_models)) {
  rpart.plot(rpart_models[[i]],
             main = paste("maxnodes ≈", nodes_list[i]),
             type = 2,
             box.col = "deepskyblue",
             fallen.leaves = TRUE
             )

 }





