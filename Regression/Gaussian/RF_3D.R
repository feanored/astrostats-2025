# -------------------------------------------------------------
# Carregar pacotes
# -------------------------------------------------------------
library(randomForest)
library(plot3D)

# -------------------------------------------------------------
# Simular dados
# -------------------------------------------------------------
set.seed(42)
n <- 400
x1 <- runif(n, -3, 3)
x2 <- runif(n, -5, 5)
y <- x1^2 + cos(x2^2) - 4*x1 - 0.5*x2 + rnorm(n, 0, 0.1)
df <- data.frame(x1 = x1, x2 = x2, y = y)

# Grid para superfície
x1_seq <- seq(-3, 3, length.out = 50)
x2_seq <- seq(-5, 5, length.out = 50)
grid <- expand.grid(x1 = x1_seq, x2 = x2_seq)

# -------------------------------------------------------------
# Função auxiliar para RF com pontos observados
# -------------------------------------------------------------
plot_rf_surface <- function(maxnodes, col = viridis::plasma(100)) {
  rf_fit <- randomForest(y ~ x1 + x2, data = df, ntree = 1, maxnodes = maxnodes)
  z_pred <- matrix(predict(rf_fit, newdata = grid),
                   nrow = length(x1_seq), ncol = length(x2_seq))

  # Superfície estimada
  persp3D(
    x = x1_seq, y = x2_seq, z = z_pred,
    theta = 40, phi = 25, expand = 0.6,
    col = col, border = "black", ticktype = "detailed",
    xlab = "x1", ylab = "x2", zlab = "y_hat",
    colkey = FALSE,
    main = paste("Nodes =", maxnodes)
  )

  # Pontos reais
  points3D(
    x = df$x1, y = df$x2, z = df$y,
    add = TRUE, col = "cyan3", pch = 19, cex = 0.4
  )
}

# -------------------------------------------------------------
# Mosaico com diferentes níveis de complexidade
# -------------------------------------------------------------
par(mfrow = c(2, 2))
plot_rf_surface(maxnodes = 2)
plot_rf_surface(maxnodes = 5)
plot_rf_surface(maxnodes = 20)
plot_rf_surface(maxnodes = 300)
