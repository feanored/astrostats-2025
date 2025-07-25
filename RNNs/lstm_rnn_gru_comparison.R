library(keras)
library(tensorflow)
library(ggplot2)
library(randomForest)

## --- dados simulados -------------------------------------------------------
set.seed(42)
n            <- 500
time         <- seq(0, 100, length.out = n)
pulse_time   <- sample(time, 1)
brightness   <- 1 +
  0.4 * sin(2 * pi * time / 5) +                      # sinal harmônico
  ifelse(abs(time - pulse_time) < 1, 1.5, 0) +        # “pico” em 1 instante
  rnorm(n, 0, 0.1)                                    # ruído

## --- normalização ----------------------------------------------------------
brightness_sc <- scale(brightness)
ctr   <- attr(brightness_sc, "scaled:center")
scl   <- attr(brightness_sc, "scaled:scale")

## --- janelas deslizantes ---------------------------------------------------
look_back <- 30
N         <- n - look_back

X_array <- array(0, dim = c(N, look_back, 1))
Y_vec   <- numeric(N)
X_tab   <- matrix(0, nrow = N, ncol = look_back)  # p/ RF e MLP

for (i in 1:N) {
  win            <- brightness_sc[i:(i + look_back - 1)]
  X_array[i, , ] <- win
  X_tab[i, ]     <- win
  Y_vec[i]       <- brightness_sc[i + look_back]
}

## TensorFlow exige float32
X_array <- tf$constant(X_array, dtype = "float32")
Y_vec   <- tf$constant(matrix(Y_vec, ncol = 1), dtype = "float32")

## --- helper de compile & fit ----------------------------------------------
compile_and_fit <- function(model, x, y) {
  model |>
    compile(optimizer = "adam", loss = "mse") |>
    fit(x, y, epochs = 30, batch_size = 32, verbose = 0)
  model
}

## --- modelos RNN via API funcional ----------------------------------------
inp <- layer_input(shape = c(look_back, 1))

model_lstm <- keras_model(
  inp, inp |> layer_lstm(32) |> layer_dense(1)
)

model_rnn <- keras_model(
  inp, inp |> layer_simple_rnn(32) |> layer_dense(1)
)

model_gru <- keras_model(
  inp, inp |> layer_gru(32) |> layer_dense(1)
)

model_lstm <- compile_and_fit(model_lstm, X_array, Y_vec)
model_rnn  <- compile_and_fit(model_rnn,  X_array, Y_vec)
model_gru  <- compile_and_fit(model_gru,  X_array, Y_vec)

## --- MLP (tabular) ---------------------------------------------------------
model_mlp <- keras_model_sequential() |>
  layer_dense(64, activation = "relu", input_shape = look_back) |>
  layer_dense(64, activation = "relu") |>
  layer_dense(1) |>
  compile(optimizer = "adam", loss = "mse")

model_mlp |> fit(X_tab, as.numeric(Y_vec), epochs = 30,
                 batch_size = 32, verbose = 0)

## --- Random Forest ---------------------------------------------------------
model_rf <- randomForest(X_tab, as.numeric(Y_vec), ntree = 200)

## --- predições (des‐scale) -------------------------------------------------
rescale <- function(z) as.numeric(z) * scl + ctr

pred_lstm <- rescale(model_lstm |> predict(X_array))
pred_rnn  <- rescale(model_rnn  |> predict(X_array))
pred_gru  <- rescale(model_gru  |> predict(X_array))
pred_mlp  <- rescale(model_mlp  |> predict(X_tab))
pred_rf   <- rescale(predict(model_rf, X_tab))

## --- data frames p/ plot ---------------------------------------------------
time_pred <- time[(look_back + 1):n]
df_list <- list(
  data.frame(time = time_pred, brightness = pred_lstm, model = "LSTM"),
  data.frame(time = time_pred, brightness = pred_rnn,  model = "RNN"),
  data.frame(time = time_pred, brightness = pred_gru,  model = "GRU"),
  data.frame(time = time_pred, brightness = pred_mlp,  model = "MLP"),
  data.frame(time = time_pred, brightness = pred_rf,   model = "Random Forest"),
  data.frame(time = time_pred,
             brightness = 1 + 0.4 * sin(2 * pi * time_pred / 5) +
               ifelse(abs(time_pred - pulse_time) < 1, 1.5, 0),
             model = "Verdadeiro")
)
df_all  <- data.frame(time, brightness)
df_pred <- do.call(rbind, df_list)

## --- visualização ----------------------------------------------------------
ggplot() +
  geom_point(data = df_all, aes(time, brightness),
             colour = "grey80", size = 0.8) +
  geom_line(data = df_pred,
            aes(time, brightness, colour = model), linewidth = 1) +
  scale_color_manual(values = c(
    "LSTM" = "darkorange",
    "GRU" = "forestgreen",
    "RNN" = "dodgerblue",
    "MLP" = "magenta",
    "Random Forest" = "brown",
    "Verdadeiro" = "black"
  )) +
  facet_wrap(~ model, ncol = 2) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none") +
  labs(x = "Tempo", y = "Brilho")

