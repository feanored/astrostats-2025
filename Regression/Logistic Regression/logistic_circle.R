# install.packages(c("ggplot2","viridis"))  # if needed
library(ggplot2)
library(viridis)

set.seed(42)

# 1. Simulate two rings for a strongly nonlinear boundary
n_per <- 500
# angles uniform
theta <- runif(2*n_per, 0, 2*pi)
# radii: half in inner circle (r<3), half in outer annulus (4<r<6)
r_inner <- sqrt(runif(n_per, 0, 3^2))
r_outer <- sqrt(runif(n_per, 4^2, 6^2))
r <- c(r_inner, r_outer)
x1 <- r * cos(theta)
x2 <- r * sin(theta)
y  <- factor(rep(c(0,1), each = n_per))

df <- data.frame(x1 = x1, x2 = x2, y = y)

# 2. Fit polynomial logistic regression (2nd-degree + interaction)
log_mod_nl <- glm(y ~ poly(x1, 2) + poly(x2, 2) + I(x1*x2),
                  family = binomial, data = df)

# 3. Create a fine grid
x1_seq <- seq(min(df$x1)-1, max(df$x1)+1, length.out = 300)
x2_seq <- seq(min(df$x2)-1, max(df$x2)+1, length.out = 300)
grid   <- expand.grid(x1 = x1_seq, x2 = x2_seq)

# 4. Predict probability on that grid
grid$prob <- predict(log_mod_nl, newdata = grid, type = "response")

# 5. Plot
ggplot() +
  # background probability heatmap
  geom_raster(data = grid, aes(x = x1, y = x2, fill = prob), interpolate = TRUE) +
  # decision contour at p = 0.5
  geom_contour(data = grid, aes(x = x1, y = x2, z = prob),
               breaks = 0.5, color = "white", size = 1) +
  # raw data points
  geom_point(data = df, aes(x = x1, y = x2, color = y), size = 1.5, alpha = 0.6) +
  # scales & theme
  scale_fill_viridis(name = "P(y = 1)", option = "magma") +
  scale_color_manual(values = c("orange","cyan3")) +
  coord_equal() +
  theme_minimal(base_size = 14) +
  labs(
    title    = "Nonlinear (Polynomial) Logistic Regression",
    subtitle = "Concentric rings simulated; model includes x1^2, x2^2, and interaction",
    x        = "x1",
    y        = "x2"
  ) +
  theme(panel.grid = element_blank(),
        legend.position = "none")
