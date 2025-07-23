library(keras)
library(tidyverse)
library(Rtsne)
library(hdf5r)
library(EBImage)
library(fs)

# Abrir o arquivo HDF5
f <- H5File$new("Binary_2_5_dataset.h5", mode = "r")
images <- f[["images"]]
labels <- f[["labels"]][]
labels <- 1 - labels  # Corrige a inversão: 0 → 1, 1 → 0
n_samples <- length(labels)

# Mapear nomes das classes
label_names <- c("round_smooth", "barred_spiral")

# Criar diretórios
paths <- c("train", "val", "test")
for (p in paths) {
  for (cls in label_names) {
    dir_create(path(paste0("dataset/", p, "/", cls)))
  }
}

# Dividir os dados (exemplo: 70% treino, 15% val, 15% teste)
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
    img <- images[ , , , id]
    img_rgb <- aperm(img, c(2, 3, 1)) / 255
    path <- sprintf("dataset/%s/%s/%05d.png", split_name, label_str, id)
    writeImage(Image(img_rgb), path, quality = 90)

    if (i %% 500 == 0) cat(split_name, ": salva", i, "imagens\n")
  }
}

# Salvar cada partição
save_split(train_idx, "train")
save_split(val_idx,   "val")
save_split(test_idx,  "test")

f$close_all()
cat("✅ Todas as imagens foram exportadas.\n")


img_size <- c(256, 256)
batch_size <- 32

train_gen <- image_data_generator(rescale = 1/255)
val_gen <- image_data_generator(rescale = 1/255)

train_data <- flow_images_from_directory(
  directory = "dataset/train",
  generator = train_gen,
  target_size = img_size,
  batch_size = batch_size,
  class_mode = "binary"
)

val_data <- flow_images_from_directory(
  directory = "dataset/val",
  generator = val_gen,
  target_size = img_size,
  batch_size = batch_size,
  class_mode = "binary"
)

#--------------------------
# 2. Visualizar algumas imagens
#--------------------------

cat("Visualizando imagens RGB...\n")

batch <- generator_next(train_data)

plot_rgb_gg <- function(img, label) {
  df <- data.frame(
    x = rep(1:dim(img)[2], each = dim(img)[1]),
    y = rep(dim(img)[1]:1, times = dim(img)[2]),
    R = as.vector(img[,,1]),
    G = as.vector(img[,,2]),
    B = as.vector(img[,,3])
  )
  df$hex <- rgb(df$R, df$G, df$B)

  ggplot(df, aes(x = x, y = y,fill=hex)) +
    geom_raster() +
    scale_fill_identity() +
    coord_fixed() +
    theme_void() +
    ggtitle(label)
}

plots <- list()
for (i in 1:6) {
  img <- batch[[1]][i,,,]
  label <- ifelse(batch[[2]][i] == 0, "Round Smooth", "Barred Spiral")
  plots[[i]] <- plot_rgb_gg(img, label)
}
gridExtra::grid.arrange(grobs = plots, ncol = 3)




input <- layer_input(shape = c(256, 256, 3))

conv_1 <- input %>%
  layer_conv_2d(filters = 32, kernel_size = 3, activation = 'relu', name = "conv1") %>%
  layer_max_pooling_2d(pool_size = 2)

conv_2 <- conv_1 %>%
  layer_conv_2d(filters = 64, kernel_size = 3, activation = 'relu', name = "conv2") %>%
  layer_max_pooling_2d(pool_size = 2)

latent <- conv_2 %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu', name = "latent")

output <- latent %>%
  layer_dense(units = 1, activation = "sigmoid", name = "classifier")

model <- keras_model(input, output)

model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = 'accuracy'
)

history <- model %>% fit(
  train_data,
  steps_per_epoch = train_data$samples %/% batch_size,
  epochs = 5,
  validation_data = val_data,
  validation_steps = val_data$samples %/% batch_size
)


activation_model <- keras_model(inputs = model$input,
                                outputs = get_layer(model, "conv1")$output)

sample_img <- batch[[1]][1,,, , drop = FALSE]
activations <- activation_model %>% predict(sample_img)
# Função para visualização de filtros
plot_activation <- function(activations, n_filters = 6) {
  plots <- list()
  for (i in 1:n_filters) {
    act <- activations[1,,,i]
    df <- as.data.frame(as.table(act))
    colnames(df) <- c("x", "y", "value")

    p <- ggplot(df, aes(x = x, y = y, fill = value)) +
      geom_raster() +
      scale_fill_viridis_c() +
      theme_void() +
      coord_fixed() +
      ggtitle(paste("Filtro", i))

    plots[[i]] <- p
  }
  grid.arrange(grobs = plots, ncol = 3)
}

plot_activation(activations)

compare_activations(model, batch, layer_name = "conv1", n_filters = 6,
                    index_round = 4, index_barred = 1)


compare_activations(model, batch, layer_name = "conv1", n_filters = 8)


# Verificar o tipo das imagens no batch
labels <- batch[[2]]

# Pegar a primeira imagem Round Smooth (label == 0)
index_round <- which(labels == 0)[1]
img_round <- batch[[1]][index_round,,,]
visualize_filter_effect(model, img_round, filter_index = 16)

# Ou a primeira Barred Spiral (label == 1)
index_barred <- which(labels == 1)[1]
img_barred <- batch[[1]][index_barred,,,]
visualize_filter_effect(model, img_barred, filter_index = 15)




# Latent features extraction
latent_model <- keras_model(inputs = model$input, outputs = get_layer(model, "latent")$output)

latent_features <- latent_model %>% predict(batch[[1]])  # 32 imagens
labels <- batch[[2]]

cat("Executando t-SNE...\n")

set.seed(42)
tsne_out <- Rtsne(latent_features, dims = 2, perplexity = 10)

# Visualização do espaço latente
latent_df <- data.frame(
  x = tsne_out$Y[,1],
  y = tsne_out$Y[,2],
  class = factor(labels, levels = c(0,1), labels = c("round_smooth", "barred_spiral"))
)

ggplot(latent_df, aes(x = x, y = y, color = class)) +
  geom_point(alpha = 0.8, size = 3) +
  theme_minimal() +
  labs(title = "Visualização do espaço latente (t-SNE)",
       x = "Dimensão 1", y = "Dimensão 2")



library(ggimage)

dir.create("tmp_batch_images", showWarnings = FALSE)

for (i in 1:dim(batch[[1]])[1]) {
  img <- batch[[1]][i,,,]
  img_rgb <- Image(img)
  path <- sprintf("tmp_batch_images/img_%02d.png", i)
  writeImage(img_rgb, path, type = "png")  # ⬅️ necessário!
}

latent_df$image_path <- sprintf("tmp_batch_images/img_%02d.png", 1:32)

library(ggimage)

ggplot(latent_df, aes(x = x, y = y)) +
  geom_image(aes(image = image_path), size = 0.05) +
  theme_void() +
  labs(title = "Espaço latente com miniaturas")



