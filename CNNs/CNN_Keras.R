library(keras)
library(tidyverse)
library(Rtsne)

library(hdf5r)

# Abrir o arquivo HDF5
f <- H5File$new("Binary_2_5_dataset.h5", mode = "r")

# Ler as imagens e os labels
dims <- f[["images"]]$dims
labels <- f[["labels"]][]

# Fechar o arquivo
f$close_all()

# CIFAR-10
cifar <- dataset_cifar10()
x_train <- cifar$train$x / 255
y_train <- cifar$train$y
x_test <- cifar$test$x / 255
y_test <- cifar$test$y


img <- x_test[1,,,]
# Convert the array to a data frame for ggplot
img_df <- data.frame(
  x = rep(1:32, each = 32),
  y = rep(32:1, times = 32), # flip vertically for correct orientation
  R = as.vector(img[,,1]),
  G = as.vector(img[,,2]),
  B = as.vector(img[,,3])
)

# Plot RGB image
ggplot(img_df, aes(x = x, y = y)) +
  geom_raster(fill = rgb(img_df$R, img_df$G, img_df$B)) +
  coord_fixed() +
  theme_void() +
  ggtitle("Single CIFAR-10 RGB Image")



# Classes: airplanes, cars, birds, etc.
class_labels <- c("plane","car","bird","cat","deer",
                  "dog","frog","horse","ship","truck")


input <- layer_input(shape = c(32, 32, 3))

conv_1 <- input %>%
  layer_conv_2d(filters=32, kernel_size=3, activation='relu', name="conv1") %>%
  layer_max_pooling_2d(pool_size=2)

conv_2 <- conv_1 %>%
  layer_conv_2d(filters=64, kernel_size=3, activation='relu', name="conv2") %>%
  layer_max_pooling_2d(pool_size=2)

latent <- conv_2 %>%
  layer_flatten() %>%
  layer_dense(units=128, activation='relu', name="latent")

output <- latent %>%
  layer_dense(units=10, activation='softmax', name="classifier")

model <- keras_model(input, output)

model %>% compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics='accuracy'
)

model %>% fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)


# Define model for intermediate activations
activation_model <- keras_model(inputs = model$input,
                                outputs = get_layer(model, "conv1")$output)

# Predict activations for one test image
sample_image <- array_reshape(x_test[1,,,], c(1, 32, 32, 3))
activations <- predict(activation_model, sample_image)

# Visualize activations
library(gridExtra)

plot_activation <- function(activations, n_filters=9) {
  plots <- list()
  for (i in 1:n_filters) {
    act <- activations[1,,,i]
    df <- as.data.frame(as.table(act))
    colnames(df) <- c("x", "y", "value")

    p <- ggplot(df, aes(x, y, fill=value)) +
      geom_raster() +
      scale_fill_viridis_c() +
      theme_void() +
      coord_fixed() +
      ggtitle(paste("Filter", i))

    plots[[i]] <- p
  }
  grid.arrange(grobs=plots, ncol=3)
}

plot_activation(activations)


# Latent features extraction
latent_model <- keras_model(inputs=model$input, outputs=get_layer(model, "latent")$output)
latent_features <- predict(latent_model, x_test[1:2000,,,])

# t-SNE embedding
set.seed(42)
tsne_out <- Rtsne(latent_features, dims=2, perplexity=30, verbose=TRUE)

# Plot t-SNE
latent_df <- data.frame(
  x = tsne_out$Y[,1],
  y = tsne_out$Y[,2],
  class = factor(class_labels[y_test[1:2000] + 1])
)

ggplot(latent_df, aes(x=x, y=y, color=class)) +
  geom_point(alpha=0.7, size=2) +
  theme_minimal() +
  labs(title="Latent Space Visualization via t-SNE",
       x="t-SNE Dim 1", y="t-SNE Dim 2") +
  theme(legend.position="right")

