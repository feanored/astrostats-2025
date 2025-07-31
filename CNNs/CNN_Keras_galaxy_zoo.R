# Load libraries

library(keras)
use_implementation("keras")     # se estiver usando o pacote `tensorflow`, pode alternar com "tensorflow"
use_backend("tensorflow")       # assegura que está usando TF backend

library(tidyverse)
library(Rtsne)
library(hdf5r)
library(EBImage)
library(fs)
library(ggimage)
library(gridExtra)

# -------------------------------------------------------------------
# 1. Load and Export HDF5 Dataset into Image Files
# -------------------------------------------------------------------
f <- H5File$new("/Users/rd23aag/Downloads/Binary_2_5_dataset.h5", mode = "r")
images <- f[["images"]]
labels <- f[["labels"]][]
labels <- 1 - labels  # Corrige a inversão (0 → 1, 1 → 0)
n_samples <- length(labels)
label_names <- c("round_smooth", "barred_spiral")

# Cria os diretórios de treino/val/test
paths <- c("train", "val", "test")
walk(paths, function(p) {
  walk(label_names, function(cls) {
    dir_create(file.path("dataset", p, cls))
  })
})

# Divisão dos dados
set.seed(42)
indices <- sample(n_samples)
train_idx <- indices[1:floor(0.7 * n_samples)]
val_idx   <- indices[(floor(0.7 * n_samples) + 1):floor(0.85 * n_samples)]
test_idx  <- indices[(floor(0.85 * n_samples) + 1):n_samples]

save_split <- function(idx, split_name) {
  for (i in seq_along(idx)) {
    id <- idx[i]
    label <- labels[id]
    label_str <- label_names[label + 1]
    img <- images[,,,id]
    img_rgb <- aperm(img, c(2, 3, 1)) / 255
    path <- sprintf("dataset/%s/%s/%05d.png", split_name, label_str, id)
    writeImage(Image(img_rgb), path, quality = 90)
    if (i %% 500 == 0) cat(split_name, ": salva", i, "imagens\n")
  }
}

walk2(list(train_idx, val_idx, test_idx), paths, save_split)
f$close_all()
cat("\u2705 Todas as imagens foram exportadas.\n")
# -------------------------------------------------------------------
# 2. Create Generators
# -------------------------------------------------------------------
img_size <- c(256, 256)
batch_size <- 32
train_gen <- image_data_generator(rescale = 1/255)
val_gen <- image_data_generator(rescale = 1/255)

train_data <- flow_images_from_directory("dataset/train", train_gen,
                                         target_size = img_size, batch_size = batch_size,
                                         class_mode = "binary", shuffle = TRUE)
val_data <- flow_images_from_directory("dataset/val", val_gen,
                                       target_size = img_size, batch_size = batch_size,
                                       class_mode = "binary", shuffle = TRUE)

# -------------------------------------------------------------------
# 3. Visualização de imagens
# -------------------------------------------------------------------
plot_rgb_gg <- function(img, label) {
  df <- expand.grid(x = 1:dim(img)[2], y = dim(img)[1]:1)
  df$R <- as.vector(img[,,1])
  df$G <- as.vector(img[,,2])
  df$B <- as.vector(img[,,3])
  df$hex <- rgb(df$R, df$G, df$B)
  ggplot(df, aes(x = x, y = y, fill = hex)) +
    geom_raster() + scale_fill_identity() + coord_fixed() + theme_void() +
    ggtitle(label)
}

batch <- generator_next(train_data)
plots <- map(1:6, ~ plot_rgb_gg(batch[[1]][.x,,,],
                                ifelse(batch[[2]][.x] == 0, "Round Smooth", "Barred Spiral")))
grid.arrange(grobs = plots, ncol = 3)

# -------------------------------------------------------------------
# 4. Modelo
# -------------------------------------------------------------------
input <- layer_input(shape = c(256, 256, 3))
model <- input %>%
  layer_conv_2d(filters = 32, kernel_size = 3, activation = 'relu', name = "conv1") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 64, kernel_size = 3, activation = 'relu', name = "conv2") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu', name = "latent") %>%
  layer_dense(units = 1, activation = "sigmoid", name = "classifier") %>%
  keras_model(input, .)

model %>% compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')
model %>% fit(train_data, steps_per_epoch = train_data$samples %/% batch_size,
              epochs = 5, validation_data = val_data,
              validation_steps = val_data$samples %/% batch_size)

# -------------------------------------------------------------------
# 5. Extração de ativações e filtros
# -------------------------------------------------------------------
activation_model <- keras_model(inputs = model$input, outputs = get_layer(model, "conv1")$output)

plot_activation <- function(activations, n_filters = 6) {
  plots <- map(1:n_filters, function(i) {
    df <- as.data.frame(as.table(activations[1,,,i]))
    colnames(df) <- c("x", "y", "value")
    ggplot(df, aes(x = x, y = y, fill = value)) +
      geom_raster() + scale_fill_viridis_c() + theme_void() + coord_fixed() +
      ggtitle(paste("Filtro", i))
  })
  grid.arrange(grobs = plots, ncol = 3)
}

activations <- activation_model %>% predict(batch[[1]][1,,, , drop = FALSE])
plot_activation(activations)

# -------------------------------------------------------------------
# 6. Espaço Latente (t-SNE) com miniaturas
# -------------------------------------------------------------------
latent_model <- keras_model(inputs = model$input, outputs = get_layer(model, "latent")$output)
latent_features <- latent_model %>% predict(batch[[1]])
labels <- batch[[2]]

set.seed(42)
tsne_out <- Rtsne(latent_features, dims = 2, perplexity = 10)
latent_df <- data.frame(
  x = tsne_out$Y[,1],
  y = tsne_out$Y[,2],
  class = factor(labels, levels = c(0,1), labels = c("round_smooth", "barred_spiral"))
)


# Miniaturas
thumb_dir <- "tmp_thumbs"                         # pick one name
dir.create(thumb_dir, recursive = TRUE, showWarnings = FALSE)

thumbs <- file.path(thumb_dir,
                    sprintf("img_%05d.png", seq_len(total)))

for (i in seq_len(total)) {
  img_rgb <- Image(images[i,,,])                  # 0‑1 numeric array
  colorMode(img_rgb) <- Color
  writeImage(img_rgb, thumbs[i], quality = 90)    # now succeeds
}


latent_df$image_path <- sprintf("tmp_batch_images/img_%02d.png", 1:dim(batch[[1]])[1])

ggplot(latent_df, aes(x = x, y = y)) +

  geom_point(aes(color = class), shape = 22, stroke = 1.5,
             size = 21, fill = NA) +
  geom_image(aes(image = image_path), size = 0.09) +
  scale_color_manual(values = c("#66c2a5", "#fc8d62"),
                     name="") +
  guides(color = guide_legend(override.aes = list(alpha = 1, size = 5))) +
  theme_xkcd() +
  theme(legend.position = "bottom") +
 # coord_cartesian(xlim=c(-100,100),ylim=c(-200,200)) +
  xlab("Latent1") + ylab("Latent2")
