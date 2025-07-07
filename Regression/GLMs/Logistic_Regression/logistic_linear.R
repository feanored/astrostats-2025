# install.packages(c("ggplot2","viridis","MASS"))  # if needed
library(ggplot2)
library(viridis)

set.seed(123)

# 1. Simulate two clusters for clear separation
n_per <- 150
# Class 0 centered at (2,2)
x1_0 <- rnorm(n_per, mean = 2, sd = 1)
x2_0 <- rnorm(n_per, mean = 2, sd = 1)
y0   <- rep(0, n_per)
# Class 1 centered at (6,6)
x1_1 <- rnorm(n_per, mean = 6, sd = 1)
x2_1 <- rnorm(n_per, mean = 6, sd = 1)
y1   <- rep(1, n_per)

df <- data.frame(
  x1 = c(x1_0, x1_1),
  x2 = c(x2_0, x2_1),
  y  = factor(c(y0, y1))
)

# 2. Fit logistic regression with both predictors
log_mod2 <- glm(y ~ x1 + x2, family = binomial, data = df)

# 3. Create a grid over the covariate space
x1_seq <- seq(min(df$x1)-1, max(df$x1)+1, length.out = 200)
x2_seq <- seq(min(df$x2)-1, max(df$x2)+1, length.out = 200)
grid   <- expand.grid(x1 = x1_seq, x2 = x2_seq)

# 4. Predict probability on that grid
grid$prob <- predict(log_mod2, newdata = grid, type = "response")

# 5. Plot
ggplot() +
  # background probability heatmap
  geom_raster(data = grid, aes(x = x1, y = x2, fill = prob), interpolate = TRUE) +
  # decision contour at p = 0.5
  geom_contour(data = grid, aes(x = x1, y = x2, z = prob),
               breaks = 0.5, color = "white", size = 1) +
  # raw data points
  geom_point(data = df, aes(x = x1, y = x2, color = y), size = 2, alpha = 0.8) +
  # scales & theme
  scale_fill_viridis(name = "P(y = 1)", option = "viridis") +
  scale_color_manual(values = c("orange","cyan3")) +
  coord_equal() +
  theme_minimal(base_size = 14) +
  labs(
    title    = "Bivariate Logistic Regression",
    subtitle = "Background colored by predicted P(y = 1)",
    x        = "Covariate x1",
    y        = "Covariate x2"
  ) +
  theme(panel.grid = element_blank(),
        legend.position="none")

