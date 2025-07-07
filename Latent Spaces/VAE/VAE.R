#––– 0) Libraries
library(keras)
library(RDRToolbox)
library(ggplot2)
library(viridis)

#––– 1) Generate Swiss Roll
set.seed(42)
swiss <- SwissRoll(2000, plot = FALSE)
x_raw <- swiss
t_est <- sqrt(x_raw[,1]^2 + x_raw[,3]^2)   # "true" roll coordinate
x_scaled <- scale(x_raw)

#––– 2) Sampling function
sampling <- function(args) {
  z_mean <- args[[1]]
  z_log_var <- args[[2]]
  epsilon <- k_random_normal(shape = k_shape(z_mean), mean = 0., stddev = 1)
  z_mean + k_exp(0.5 * z_log_var) * epsilon
}

#––– 3) Encoder
latent_dim <- 2
input_dim <- ncol(x_scaled)

inputs <- layer_input(shape = input_dim, name = "encoder_input")

h <- inputs %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(64,  activation = "relu")

z_mean <- h %>% layer_dense(latent_dim, name = "z_mean")
z_log_var <- h %>% layer_dense(latent_dim, name = "z_log_var")

z <- list(z_mean, z_log_var) %>%
  layer_lambda(sampling, name = "z")

encoder <- keras_model(inputs, list(z_mean, z_log_var, z))

#––– 4) Decoder
decoder_input <- layer_input(shape = latent_dim, name = "decoder_input")

decoder_output <- decoder_input %>%
  layer_dense(64,  activation = "relu") %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(input_dim, activation = "linear")

decoder <- keras_model(decoder_input, decoder_output)

#––– 5) VAE Model
z_vals <- encoder(inputs)
reconstruction <- decoder(z_vals[[3]])

vae <- keras_model(inputs, reconstruction)

#––– 6) Custom loss function
vae_loss <- function(x, x_decoded) {
  rec <- k_sum(k_square(x - x_decoded), axis = -1L)
  kl  <- -0.5 * k_sum(1 + z_vals[[2]] - k_square(z_vals[[1]]) - k_exp(z_vals[[2]]), axis = -1L)
  k_mean(rec + kl)
}

vae %>% compile(optimizer = "adam", loss = vae_loss)

#––– 7) Train
vae %>% fit(
  x_scaled, x_scaled,
  epochs = 200,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 0
)

#––– 8) Get latent space (z_mean)
encoded <- predict(encoder, x_scaled)
z_mean_vals <- encoded[[1]]

vae_df <- data.frame(
  Lat1 = z_mean_vals[,1],
  Lat2 = z_mean_vals[,2],
  roll = t_est
)

#––– 9) Plot
ggplot(vae_df, aes(Lat1, Lat2, color = roll)) +
  geom_point(size = 1.5) +
  scale_color_viridis_c() +
  theme_minimal() +
  ggtitle("VAE Latent Space — Swiss Roll")
