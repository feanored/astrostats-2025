# ==== Novos kernels intuitivos ====
img <- as.cimg(matrix(0, nrow = 100, ncol = 100))

# Desenhar um quadrado branco no centro
img[40:60, 40:60] <- 1  # valor 1 = branco



img <- load.image("NGC628.png") %>% grayscale()  # use your saved image path here
img <- resize(img, size_x = 512, size_y = 512)     # Resize for simplicity





# Blur (Box filter) — suaviza
kernel_gaussian <- as.cimg(matrix(c(1, 2, 1,
                                2, 4, 2,
                                1, 2, 1) / 16, nrow = 3, byrow = TRUE))

# Laplaciano — bordas em todas as direções
kernel_laplacian <- as.cimg(matrix(c(0,  1, 0,
                                     1, -4, 1,
                                     0,  1, 0), nrow=3, byrow=TRUE))

# Emboss — efeito de relevo/direcionalidade
kernel_emboss <- as.cimg(matrix(c(-2, -1, 0,
                                  -1,  1, 1,
                                  0,  1, 2), nrow=3, byrow=TRUE))

# Sobel X e Y (mais explícitos)
kernel_sobel_x <- as.cimg(matrix(c(-1, 0, 1,
                                   -2, 0, 2,
                                   -1, 0, 1), nrow = 3, byrow = TRUE))

kernel_sobel_y <- as.cimg(matrix(c(-1, -2, -1,
                                   0,  0,  0,
                                   1,  2,  1), nrow = 3, byrow = TRUE))



plot(kernel_blur)
plot(kernel_laplacian)
plot(kernel_sobel_x)
plot(kernel_sobel_y)

# ==== Organizar com patchwork ====
(p1 | p2 | p3) / (p4 | p5 | p6)

# ==== Aplicar todos os filtros ====
img_blur     <- correlate(img, kernel_blur)
img_lapl     <- correlate(img, kernel_laplacian)
img_emboss   <- correlate(img, kernel_emboss)
img_sobel_x  <- correlate(img, kernel_sobel_x)
img_sobel_y  <- correlate(img, kernel_sobel_y)

# ==== Converter para data.frames ====
df_img      <- as.data.frame(img)
df_blur     <- as.data.frame(img_blur)
df_lapl     <- as.data.frame(img_lapl)
df_emboss   <- as.data.frame(img_emboss)
df_sobelx   <- as.data.frame(img_sobel_x)
df_sobely   <- as.data.frame(img_sobel_y)

# ==== Plots ====
p1 <- plot_img(df_img,     "Original")
p2 <- plot_img(df_blur,    "Blur")
p3 <- plot_img(df_lapl,    "Laplacian")
p4 <- plot_img(df_emboss,  "Emboss")
p5 <- plot_img(df_sobelx,  "Sobel X")
p6 <- plot_img(df_sobely,  "Sobel Y")

# ==== Organizar com patchwork ====
(p1 | p2 | p3) / (p4 | p5 | p6)
