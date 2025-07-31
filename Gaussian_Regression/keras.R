library(keras)
#install_keras(
#  method = "virtualenv",
#  envname = "r-keras-ok",
#  tensorflow = "2.14.0",
#  extra_packages = c("numpy<2", "pillow", "scipy", "pandas")
#)
reticulate::use_virtualenv("r-keras-ok", required = TRUE)

# Dados
set.seed(42)
n <- 1000
x <- runif(n, 0, 6)
y <- 1 - 3 * x + 8 * sin(x^2) + rnorm(n, 0, 1)

# Normalização
x_scaled <- scale(x)
y_scaled <- scale(y)

# Define entrada
input <- layer_input(shape = 1)

# Rede
output <- input %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1)

# Modelo funcional
model <- keras_model(inputs = input, outputs = output)

model %>% compile(
  loss = "mse",
  optimizer = optimizer_adam(),
  metrics = list("mae")
)

# Ajuste
history <- model %>% fit(
  x = x_scaled,
  y = y_scaled,
  epochs = 1000,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 0
)

# Predição
x_grid <- seq(0, 6, length.out = 200)
x_grid_scaled <- scale(x_grid, center = attr(x_scaled, "scaled:center"), scale = attr(x_scaled, "scaled:scale"))
y_pred <- model %>% predict(x_grid_scaled)

# Retornando à escala original
y_pred <- y_pred * attr(y_scaled, "scaled:scale") + attr(y_scaled, "scaled:center")

# Visualização
plot(x, y, pch = 19, col = "gray80", main = "MLP via Keras")
lines(x_grid, y_pred, col = "orange3", lwd = 3)
lines(xx, 1 - 3 * xx + 8 * sin(xx^2), col = 'blue', lwd = 2)     # Linha verdadeira



df <- data.frame(x = x, y = y)
df_pred <- data.frame(x = x_grid, y = y_pred)
df_true <- data.frame(x = x_grid, y = 1 - 3 * x_grid + 8 * sin(x_grid^2))

ggplot() +
  geom_point(data = df, aes(x = x, y = y), color = "gray80", shape = 19) +
  geom_line(data = df_pred, aes(x = x, y = y), color = "orange3", linewidth = 1.5) +
  geom_line(data = df_true, aes(x = x, y = y), color = "blue", linewidth = 1, linetype = "solid") +
  labs(title = "MLP via Keras", x = "x", y = "y") +
  theme_xkcd()
