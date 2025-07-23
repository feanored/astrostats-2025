# =============================== #
#   AUTOENCODER 2D vs PCA – Galáxias
# =============================== #
library(keras3)
library(tidyverse)
library(ggimage)
library(EBImage)
library(abind)
library(fs)

# ----------- CONFIGURAÇÕES -----------
img_size     <- c(64, 64)
latent_dim   <- 128
n_per_class  <- 64
train_dir    <- "dataset/train"
thumb_dir    <- "tmp_thumbs"
dir_create(thumb_dir, recurse = TRUE)

# ----------- MODELO AUTOENCODER -----------
input_img <- layer_input(shape = c(img_size, 3))

# Codificador
encoded <- input_img |>
  layer_conv_2d(32, 3, padding = "same") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu() |>
  layer_max_pooling_2d(pool_size = 2) |>

  layer_conv_2d(64, 3, padding = "same") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu() |>
  layer_max_pooling_2d(pool_size = 2) |>

  layer_conv_2d(128, 3, padding = "same") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu() |>
  layer_max_pooling_2d(pool_size = 2) |>

  layer_conv_2d(256, 3, padding = "same") |>  # Camada mais profunda
  layer_batch_normalization() |>
  layer_activation_leaky_relu() |>
  layer_flatten() |>
  layer_dense(256, activation = "relu") |>
  layer_dense(latent_dim, name = "latent")

encoder <- keras_model(input_img, encoded)

# Decodificador
latent_input <- layer_input(shape = latent_dim)


decoded <- latent_input |>
  layer_dense((img_size[1]/8)*(img_size[2]/8)*256, activation = "relu") |>
  layer_reshape(c(img_size[1]/8, img_size[2]/8, 256)) |>

  layer_conv_2d_transpose(128, 3, strides = 2, padding = "same") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu() |>

  layer_conv_2d_transpose(64, 3, strides = 2, padding = "same") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu() |>

  layer_conv_2d_transpose(32, 3, strides = 2, padding = "same") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu() |>

  layer_conv_2d(3, 3, activation = "sigmoid", padding = "same")


decoder <- keras_model(latent_input, decoded)

#ssim_loss <- function(y_true, y_pred) {
#  1 - tf$image$ssim(y_true, y_pred, max_val = 1.0)
#}


# Modelo completo
auto_out <- decoder(encoder(input_img))
autoencoder <- keras_model(input_img, auto_out)
autoencoder$compile(optimizer = "adam", loss = "mse")

# ----------- GERADOR PARA TREINAMENTO -----------
train_imgs <- flow_images_from_directory(
  directory   = train_dir,
  generator   = image_data_generator(rescale = 1/255),
  target_size = img_size,
  batch_size  = 32,
  class_mode  = "input",  # X = Y
  shuffle     = TRUE
)

autoencoder$fit(
  x = train_imgs,
  epochs = as.integer(6),
  steps_per_epoch = as.integer(ceiling(train_imgs$samples/32)),
  verbose = as.integer(2)
)


set.seed(42)  # Para reprodutibilidade
# ----------- AMOSTRA BALANCEADA (16+16 imagens) -----------
paths_round  <- sort(dir_ls(file.path(train_dir, "round_smooth"), glob = "*.png"))[1:n_per_class]
paths_barred <- sort(dir_ls(file.path(train_dir, "barred_spiral"), glob = "*.png"))[1:n_per_class]
sample_paths <- c(paths_round, paths_barred)

# Carrega e empilha as imagens mantendo ordem
x_batch <- abind::abind(
  lapply(sample_paths, \(p) {
    img <- image_load(p, target_size = img_size) |>
      image_to_array() / 255
    array_reshape(img, c(1, dim(img)))  # Batch dim
  }),
  along = 1
)

labels_b <- factor(
  c(rep("round_smooth", n_per_class), rep("barred_spiral", n_per_class)),
  levels = c("round_smooth", "barred_spiral")
)

# ----------- LATENTES: Autoencoder e PCA -----------
ae_latent  <- encoder %>% predict(x_batch)
pca_latent <- prcomp(array_reshape(x_batch, c(nrow(x_batch), prod(img_size) * 3)))$x[,1:2]
tsne_ae   <- Rtsne::Rtsne(ae_latent, dims = 2,perplexity = 5)$Y

# ----------- MINIATURAS -----------
thumb_paths <- file.path(thumb_dir, sprintf("img_%02d.png", seq_along(sample_paths)))

for (i in seq_along(sample_paths)) {
  img_rgb <- Image(x_batch[i,,,])
  colorMode(img_rgb) <- Color
  writeImage(img_rgb, thumb_paths[i], quality = 90)
}

# ----------- DATAFRAME PARA PLOT -----------
make_df <- function(mat, method)
  tibble(x = mat[,1], y = mat[,2],
         class = labels_b,
         image = thumb_paths,
         method = method)

plot_df <- bind_rows(
#  make_df(ae_latent,  "Autoencoder"),
  make_df(pca_latent, "PCA"),
  make_df(tsne_ae, "Autoencoder")
)

# ----------- PLOT -----------
ggplot(plot_df, aes(x, y)) +
  geom_point(aes(color = class),
             shape = 22, size = 20, stroke = 1.4, fill = NA) +
  geom_image(aes(image = image), size = 0.09) +
  scale_color_manual(values = c("#66c2a5", "#fc8d62"), name = "") +
  facet_wrap(~method, scales = "free") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  labs(title = "Galaxy latent space – Autoencoder vs PCA",
       x = "Latent dim 1", y = "Latent dim 2")





# ----------- PREVISÃO DO AUTOENCODER -----------
recons <- autoencoder %>% predict(x_batch)

# Flatten das imagens (como no PCA que você já aplicou)
# PCA em cima das imagens achatadas
# Verifique a dimensão original
dim_x <- dim(x_batch)  # N, H, W, C

# Reformatar para [N, H * W * C]
x_flat <- array_reshape(x_batch, c(dim_x[1], prod(dim_x[2:4])))

# PCA
pca_model <- prcomp(x_flat, center = TRUE)
x_proj    <- pca_model$x[, 1:latent_dim]
x_recon   <- x_proj %*% t(pca_model$rotation[, 1:latent_dim])

# Recentrar
x_recon <- scale(x_recon, center = -pca_model$center, scale = FALSE)

# Reconstruir como [N, H, W, C]
pca_imgs <- array_reshape(x_recon, dim = dim_x)



# Função para salvar thumbnails
save_thumbs <- function(img_array, prefix, dir = "tmp_thumbs") {
  files <- file.path(dir, sprintf("%s_%02d.png", prefix, seq_len(dim(img_array)[1])))
  for (i in seq_along(files)) {
    img <- img_array[i,,,]
    img_rgb <- Image(img, colormode = "Color")  # Define explicitamente como imagem RGB
    writeImage(img_rgb, files[i], quality = 90)
  }
  files
}

# Salva miniaturas
orig_imgs <- save_thumbs(x_batch,          "orig")
ae_imgs   <- save_thumbs(recons,           "ae")
pca_imgs  <- save_thumbs(pca_imgs, "pca")


# Número de imagens a mostrar
n_show <- 5

# Índices aleatórios (mesmos para todas as versões)
set.seed(123)  # para reprodutibilidade
sel_idx <- sample(seq_along(orig_imgs), n_show)

# Correto: cada id é repetido 3 vezes (Original, AE, PCA)
df_plot <- tibble(
  id    = rep(1:n_show, times = 3),
  tipo  = rep(c("Original", "Autoencoder", "PCA"), each = n_show),
  image = c(orig_imgs[sel_idx],
            ae_imgs[sel_idx],
            pca_imgs[sel_idx])
)


ggplot(df_plot, aes(x = factor(id), y = tipo)) +
  geom_image(aes(image = image), size = 0.3) +
  theme_minimal(base_size = 12) +
  labs(x = NULL, y = NULL) +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_text(size = 10),
    panel.grid  = element_blank(),
    panel.spacing = unit(0.1, "lines"),    # <-- chave para reduzir espaço entre linhas
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
  )



