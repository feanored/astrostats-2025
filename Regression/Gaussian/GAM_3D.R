# -------------------------------------------------------------
# Carregar pacotes
# -------------------------------------------------------------
library(mgcv)
library(plot3D)

# -------------------------------------------------------------
# Simular dados
# -------------------------------------------------------------
set.seed(42)
n <- 400
x1 <- runif(n, -3, 3)
x2 <- runif(n, -3, 3)
y <- x1^2 + cos(x2^2) - 4*x1 - 0.5*x2 + rnorm(n, 0, 0.1)
df <- data.frame(x1 = x1, x2 = x2, y = y)

# Grid para superfície
x1_seq <- seq(-3, 3, length.out = 50)
x2_seq <- seq(-3, 3, length.out = 50)
grid <- expand.grid(x1 = x1_seq, x2 = x2_seq)

# -------------------------------------------------------------
# Função auxiliar com pontos observados
# -------------------------------------------------------------
plot_gam_surface <- function(k, col = viridis::plasma(100)) {
  gam_fit <- gam(y ~ s(x1, x2, k = k), data = df)
  z_pred <- matrix(predict(gam_fit, newdata = grid),
                   nrow = length(x1_seq), ncol = length(x2_seq))

  # Plot da superfície
  persp3D(
    x = x1_seq, y = x2_seq, z = z_pred,
    theta = 40, phi = 25, expand = 0.6,
    col = col, border = "black", ticktype = "detailed",
    xlab = "x1", ylab = "x2", zlab = "y_hat",
    main = paste("GAM - k =", k),
    colkey = FALSE
  )

  # Adiciona os pontos observados
  points3D(
    x = df$x1, y = df$x2, z = df$y,
    add = TRUE, col = "cyan2", pch = 19, cex = 0.4
  )
}

# -------------------------------------------------------------
# Mosaico com diferentes suavizações
# -------------------------------------------------------------
par(mfrow = c(2, 2))
plot_gam_surface(k = 5)
plot_gam_surface(k = 10)
plot_gam_surface(k = 20)
plot_gam_surface(k = 50)

