# install.packages(c("nnet", "ggplot2", "viridis"))  # if needed
library(nnet)
library(ggplot2)
library(viridis)

set.seed(42)

# 1. Simulate two rings for a strongly nonlinear boundary
n_per <- 500
theta <- runif(2 * n_per, 0, 2 * pi)
r_inner <- sqrt(runif(n_per, 0, 3^2))
r_outer <- sqrt(runif(n_per, 4^2, 6^2))
r <- c(r_inner, r_outer)
x1 <- r * cos(theta)
x2 <- r * sin(theta)
y <- factor(rep(c(0, 1), each = n_per))

df <- data.frame(x1 = x1, x2 = x2, y = y)

# 2. Fit neural network on raw features with one hidden layer
# Use class.ind to encode target for nnet
X <- as.matrix(df[, c("x1", "x2")])
Y <- class.ind(df$y)  # one-hot encoding (2 columns)

nn_mod <- nnet(
  x = X,
  y = Y,
  size = 10,            # 10 hidden units (can adjust)
  softmax = TRUE,       # for classification
  maxit = 300,
  decay = 1e-4,
  trace = FALSE
)

# 3. Create a fine grid
x1_seq <- seq(min(df$x1) - 1, max(df$x1) + 1, length.out = 300)
x2_seq <- seq(min(df$x2) - 1, max(df$x2) + 1, length.out = 300)
grid <- expand.grid(x1 = x1_seq, x2 = x2_seq)
grid_matrix <- as.matrix(grid)

# 4. Predict probabilities on the grid
grid$prob <- as.vector(predict(nn_mod, grid_matrix)[, 2])  # P(y = 1)

# 5. Plot
ggplot() +
  geom_raster(data = grid, aes(x = x1, y = x2, fill = prob), interpolate = TRUE) +
  geom_contour(data = grid, aes(x = x1, y = x2, z = prob),
               breaks = 0.5, color = "white", size = 1) +
  geom_point(data = df, aes(x = x1, y = x2, color = y), size = 1.5, alpha = 0.6) +
  scale_fill_viridis(name = "P(y = 1)", option = "magma") +
  scale_color_manual(values = c("orange", "cyan3")) +
  coord_equal() +
  theme_minimal(base_size = 14) +
  labs(
    title = "Nonlinear MLP via nnet",
    subtitle = "Hidden layer with 10 units, trained on raw x1, x2",
    x = "x1", y = "x2"
  ) +
  theme(panel.grid = element_blank(),
        legend.position = "none")
