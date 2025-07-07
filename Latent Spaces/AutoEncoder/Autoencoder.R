# Load packages
library(ggplot2)
library(keras)  # Using keras instead of keras3
library(gridExtra)
library(RDRToolbox)

set.seed(42)
swiss   <- SwissRoll(2000)
x_raw   <- swiss               # 500×3
t_est  <- sqrt(x_raw[,1]^2 + x_raw[,3]^2)


#––– Preprocess
x_scaled <- scale(x_raw)  # zero‐center & unit variance


# 4) PCA
pca_res <- prcomp(x_scaled)
pca_df  <- data.frame(PC1 = pca_res$x[,1],
                      PC2 = pca_res$x[,2],
                      roll = t_est )

#––– Deterministic AE
inp <- layer_input(shape = 3, name = "encoder_input")

encoder_model <- inp %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(64,  activation = "relu") %>%
  layer_dense(2,   activation = "linear", name = "bottleneck") %>%
  keras_model(inputs = inp, outputs = .)

#—— Decoder (sequential, simpler API) ——
decoder <- keras_model_sequential(name = "decoder") %>%
  layer_dense(64,  activation = "relu", input_shape = 2) %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(3,   activation = "linear")


ae <- keras_model(
  inputs  = inp,
  outputs = decoder(encoder_model(inp)),
  name    = "autoencoder"
)

ae %>% compile(optimizer = "adam", loss = "mse")

ae %>% fit(
  x_scaled, x_scaled,
  epochs           = 200,
  batch_size       = 32,
  validation_split = 0.2,
  verbose          = 0
)


latent <- encoder_model %>% predict(x_scaled)
ae_df  <- data.frame(Lat1 = latent[,1],
                     Lat2 = latent[,2],
                     roll = t_est)


# 6) Plot side by side
p1 <- ggplot(pca_df, aes(PC1, PC2, color=roll)) +
  geom_point(size=2) + scale_color_viridis_c(option="B")  +
  ggtitle("PCA on Swiss Roll") + theme_minimal()

p2 <- ggplot(ae_df, aes(Lat1, Lat2, color=roll)) +
  geom_point(size=2) + scale_color_viridis_c(option="B") +
  ggtitle("AE Bottleneck") + theme_minimal()




grid.arrange(p1, p2, ncol=2)



library(scatterplot3d)

# Basic static 3D scatter
scatterplot3d(
  x_raw[,1], x_raw[,2], x_raw[,3],
  color = viridis::viridis(2000,option="B")[rank(t_est)],
  pch = 19,
  xlab = "X", ylab = "Y", zlab = "Z",
  main = "Swiss Roll in 3D"
)


library(rgl)

# Open an rgl device
plot3d(
  x_raw[,1], x_raw[,2], x_raw[,3],
  col = viridis::viridis(2000)[rank(t_est)],
  size = 5,
  xlab = "X", ylab = "Y", zlab = "Z"
)
