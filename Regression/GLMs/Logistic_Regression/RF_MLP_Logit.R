# install.packages(c("ggplot2", "viridis", "nnet", "randomForest"))  # if needed
library(ggplot2)
library(viridis)
library(nnet)
library(randomForest)

#--------------------------
# 1. Spiral Data Generator
#--------------------------
make_spiral <- function(n_points = 500, noise = 0.2, n_turns = 1.5) {
  theta <- sqrt(runif(n_points)) * 2 * pi * n_turns
  r1 <- theta
  r2 <- theta + pi
  x1 <- r1 * cos(theta) + rnorm(n_points, 0, noise)
  y1 <- r1 * sin(theta) + rnorm(n_points, 0, noise)
  x2 <- r2 * cos(theta) + rnorm(n_points, 0, noise)
  y2 <- r2 * sin(theta) + rnorm(n_points, 0, noise)

  df <- data.frame(
    x1 = c(x1, x2),
    x2 = c(y1, y2),
    class = factor(c(rep(0, n_points), rep(1, n_points)))
  )
  return(df)
}

# Generate the spiral data
set.seed(42)
df <- make_spiral(n_points = 500, noise = 0.2, n_turns = 1.5)

#--------------------------
# 2. Build prediction grid
#--------------------------
x1_seq <- seq(min(df$x1) - 1, max(df$x1) + 1, length.out = 300)
x2_seq <- seq(min(df$x2) - 1, max(df$x2) + 1, length.out = 300)
grid <- expand.grid(x1 = x1_seq, x2 = x2_seq)

#--------------------------
# 3. GLM baseline
#--------------------------
glm_mod <- glm(class ~ x1 + x2, family = binomial, data = df)
grid$prob_glm <- predict(glm_mod, newdata = grid, type = "response")

#--------------------------
# 4. Neural Net (MLP)
#--------------------------
nn_mod <- nnet(
  x = df[, c("x1", "x2")],
  y = class.ind(df$class),
  size = 25,          # hidden layer size
  softmax = TRUE,     # for classification
  maxit = 300,
  decay = 1e-4,
  trace = FALSE
)
grid$prob_nn <- predict(nn_mod, newdata = grid)[, 2]

#--------------------------
# 5. Random Forest
#--------------------------
rf_mod <- randomForest(class ~ x1 + x2, data = df, ntree = 1000)
grid$prob_rf <- as.numeric(predict(rf_mod, newdata = grid, type = "prob")[, 2])

#--------------------------
# 6. Plot function
#--------------------------
plot_model <- function(grid, prob_col, df, title) {
  ggplot() +
    geom_raster(data = grid, aes(x = x1, y = x2, fill = !!as.name(prob_col)), interpolate = TRUE) +
    geom_contour(data = grid, aes(x = x1, y = x2, z = !!as.name(prob_col)),
                 breaks = 0.5, color = "white", size = 0.8) +
    geom_point(data = df, aes(x = x1, y = x2, color = class), size = 1.5, alpha = 0.6) +
    scale_fill_viridis(name = "P(class=1)", option = "magma") +
    scale_color_manual(values = c("orange", "cyan3")) +
    coord_equal() +
    labs(title = title, x = "x1", y = "x2") +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none", panel.grid = element_blank())
}

#--------------------------
# 7. Visualize
#--------------------------
plot_glm <- plot_model(grid, "prob_glm", df, "GLM (Linear)")
plot_nn  <- plot_model(grid, "prob_nn", df, "Neural Net (MLP, 10 hidden units)")
plot_rf  <- plot_model(grid, "prob_rf", df, "Random Forest (200 trees)")

# Display
print(plot_glm)
print(plot_nn)
print(plot_rf)
