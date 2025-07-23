# -------------------------------------------------------------
# Load Required Libraries
# -------------------------------------------------------------
library(mgcv)
library(ggplot2)
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
# 2. Experimento: GAM com diferentes valores de k
# -------------------------------------------------------------
k_list <- c(3, 5, 10, 20, 50, 100)
gam_results <- list()

for (k_val in k_list) {
  label <- paste0("k = ", k_val)
  gam_fit <- gam(y ~ s(x1, k = k_val), data = df, method = "REML")
  ypred <- predict(gam_fit, newdata = data.frame(x1 = xx))

  gam_results[[label]] <- data.frame(
    xx = xx,
    ypred = ypred,
    model = label
  )
}

# -------------------------------------------------------------
# 3. Visualização: GAMs
# -------------------------------------------------------------
df_obs <- data.frame(x1 = x1, y = y)
df_pred <- do.call(rbind, gam_results)
df_pred$model <- factor(df_pred$model, levels = paste0("k = ", k_list))

plot_gam <- ggplot() +
  geom_point(data = df_obs, aes(x = x1, y = y), color = "gray30", alpha = 0.5) +
  geom_line(data = df_pred, aes(x = xx, y = ypred), size = 1, color = "darkgreen") +
  facet_wrap(~ model, ncol = 2) +
  labs(
    title = "Generalized Additive Models",
    x = "x1", y = "y"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

print(plot_gam)
