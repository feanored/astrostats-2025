# Required packages
if (!requireNamespace("imager")) install.packages("imager")
library(imager)

# Load and convert image
img <- load.image("NGC628.png") %>% grayscale()  # use your saved image path here
img <- resize(img, size_x = 512, size_y = 512)     # Resize for simplicity

# Gradientes sobel
grad <- imgradient(img, "xy")
grad$mag <- sqrt(grad$x^2 + grad$y^2)

# Converter para data frames
df_img  <- as.data.frame(img)
df_dx   <- as.data.frame(grad$x)
df_dy   <- as.data.frame(grad$y)
df_mag  <- as.data.frame(grad$mag)

# Função genérica para plotar imagens com ggplot2
plot_img <- function(df, title = "Imagem", low = "black", high = "white") {
  ggplot(df, aes(x, y, fill = value)) +
    geom_raster() +
    scale_y_reverse() +  # Corrigir a orientação da imagem
    scale_fill_gradient(low = low, high = high) +
    coord_equal() +
    theme_void() +
    theme(legend.position = "none",
          plot.title = element_text(hjust = 0.5)) +
    ggtitle(title)
}

# Plotar tudo em grid bonito
library(patchwork)

p1 <- plot_img(df_img, "Original (gray)")
p2 <- plot_img(df_dx, "Vertical edges (dx)", low = "black", high = "orange")
p3 <- plot_img(df_dy, "Horizontal edges (dy)", low = "black", high = "deepskyblue")
p4 <- plot_img(df_mag, "Gradient Magnitude", low = "black", high = "white")

(p1 | p2) / (p3 | p4)




# Define kernels
kernel_sharpen <- as.cimg(matrix(c( 0, -1,  0,
                                    -1,  5, -1,
                                    0, -1,  0), nrow=3, byrow=TRUE))

kernel_edge <- as.cimg(matrix(c(0,  1, 0,
                                1, -4, 1,
                                0,  1, 0), nrow=3, byrow=TRUE))

kernel_strong_edge <- as.cimg(matrix(c(-1, -2, -1,
                                       0,  0,  0,
                                       1,  2,  1), nrow=3, byrow=TRUE))


# Aplicar convoluções
img_sharp  <- correlate(img, kernel_sharpen)
img_edge   <- correlate(img, kernel_edge)
img_strong <- correlate(img, kernel_strong_edge)

# Converter para data frames
df_orig   <- as.data.frame(img)
df_sharp  <- as.data.frame(img_sharp)
df_edge   <- as.data.frame(img_edge)
df_strong <- as.data.frame(img_strong)

# Plot genérico
plot_img <- function(df, title, low = "black", high = "white") {
  ggplot(df, aes(x, y, fill = value)) +
    geom_raster() +
    scale_y_reverse() +
    coord_equal() +
    scale_fill_gradient(low = low, high = high) +
    theme_void() +
    ggtitle(title) +
    theme(plot.title = element_text(hjust = 0.5),legend.position = "none")
}

# Plot em estilo lado-a-lado
p1 <- plot_img(df_orig,   "Original")
p2 <- plot_img(df_sharp,  "Sharpen",         low = "black", high = "white")
p3 <- plot_img(df_edge,   "Edge Detect",     low = "black", high = "white")
p4 <- plot_img(df_strong, "\"Strong\" Edge", low = "black", high = "white")

(p1 | p2) / (p3 | p4)





img <- load.image("NGC628.png") %>% grayscale() %>% resize(128, 128)
img_mat <- as.matrix(img[,,1,1])  # convert to plain matrix

# Define a sharpening kernel
kernel <- matrix(c(0, -1,  0,
                   -1,  5, -1,
                   0, -1,  0), nrow = 3, byrow = TRUE)

# Apply our handcrafted correlate2d
filtered <- correlate2d(img_mat, kernel)

# Plotting
image(t(apply(img_mat, 2, rev)), col = gray.colors(256), main = "Original")
image(t(apply(filtered, 2, rev)), col = gray.colors(256), main = "Filtered")


