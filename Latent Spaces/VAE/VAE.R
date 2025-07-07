library(keras3)
library(RDRToolbox)
library(ggplot2)
library(viridis)

# 1. Data
set.seed(42)
swiss <- SwissRoll(2000)
x_raw <- swiss
t_est <- sqrt(x_raw[,1]^2 + x_raw[,3]^2)
x_scaled <- scale(x_raw)
input_dim <- ncol(x_scaled)
latent_dim <- 2

# 2. Sampling function for Lambda
sampling <- function(args) {
  z_mean <- args[[1]]
  z_log_var <- args[[2]]
  epsilon <- k_random_normal(shape = k_shape(z_mean), mean = 0, stddev = 1)
  z_mean + k_exp(0.5 * z_log_var) * epsilon
}

# 3. Encoder
inputs <- layer_input(shape = input_dim, name = "encoder_input")
h <- inputs %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(64,  activation = "relu")
z_mean <- h %>% layer_dense(latent_dim, name = "z_mean")
z_log_var <- h %>% layer_dense(latent_dim, name = "z_log_var")
z <- list(z_mean, z_log_var) %>%
  layer_lambda(sampling, name = "z")
encoder <- keras_model(inputs, list(z_mean, z_log_var, z), name = "encoder")

# 4. Decoder
latent_inputs <- layer_input(shape = latent_dim, name = "z_sampling")
decoder_h <- latent_inputs %>%
  layer_dense(64,  activation = "relu") %>%
  layer_dense(128, activation = "relu")
outputs <- decoder_h %>% layer_dense(input_dim, activation = "linear")
decoder <- keras_model(latent_inputs, outputs, name = "decoder")

# 5. VAE Model (symbolic tensors)
z_mean_out <- encoder(inputs)[[1]]
z_log_var_out <- encoder(inputs)[[2]]
z_sampled <- encoder(inputs)[[3]]
reconstruction <- decoder(z_sampled)

vae <- keras_model(inputs, reconstruction)

# 6. Register KL loss to the model using add_loss
kl_loss <- -0.5 * k_mean(
  1 + z_log_var_out - k_square(z_mean_out) - k_exp(z_log_var_out),
  axis = -1L
)
vae$add_loss(kl_loss)

vae %>% compile(
  optimizer = optimizer_adam(),
  loss      = "mse"   # Just the reconstruction loss; KL added with add_loss
)

vae %>% fit(
  x_scaled, x_scaled,
  epochs = 200,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 0
)

# 7. Get latent codes
z_mean_mat <- predict(encoder, x_scaled)[[1]]
vae_df <- data.frame(
  Lat1 = z_mean_mat[,1],
  Lat2 = z_mean_mat[,2],
  roll = t_est
)

# 8. Plot latent space
ggplot(vae_df, aes(Lat1, Lat2, color = roll)) +
  geom_point(size = 1.3) +
  scale_color_viridis_c() +
  theme_minimal() +
  ggtitle("VAE Latent Space (keras3, R-native)")
