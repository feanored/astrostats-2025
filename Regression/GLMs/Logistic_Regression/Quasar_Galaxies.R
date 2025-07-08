# install.packages(c("ggplot2", "viridis", "nnet", "randomForest", "uwot", "dplyr", "tidyr"))
library(ggplot2)
library(viridis)
library(nnet)
library(randomForest)
library(uwot)
library(dplyr)
library(tidyr)

#--------------------------
# 1. Dados e pré-processamento
#--------------------------
df <- read.csv("https://raw.githubusercontent.com/RafaelSdeSouza/astrostatistics-2023/refs/heads/main/solutions/galaxyquasar.csv")

df2 <- df %>%
  mutate(ug = u - g, gr = g - r, ri = r - i, iz = i - z) %>%
  drop_na(ug, gr, ri, iz) %>%
  mutate(class = factor(class))

# PCA para visualização
color_matrix <- df2 %>% select(ug, gr, ri, iz) %>% scale()
pca <- prcomp(color_matrix)

df2 <- df2 %>%
  mutate(
    PC1 = pca$x[, 1],
    PC2 = pca$x[, 2]
  ) %>%
  mutate(class = factor(class))

#--------------------------
# 1. Grid cartesiano no plano PCA
#--------------------------
x1_seq <- seq(min(df2$PC1) - 1, max(df2$PC1) + 1, length.out = 300)
x2_seq <- seq(min(df2$PC2) - 1, max(df2$PC2) + 1, length.out = 300)
grid <- expand.grid(x1 = x1_seq, x2 = x2_seq)

#--------------------------
# 2. Inversão aproximada da PCA (para 4D)
#--------------------------
rotation <- pca$rotation[, 1:2]
grid_matrix_scaled <- as.matrix(grid) %*% t(rotation)
scaling <- attr(color_matrix, "scaled:scale")
centering <- attr(color_matrix, "scaled:center")
grid_colors <- sweep(grid_matrix_scaled, 2, scaling, "*")
grid_colors <- sweep(grid_colors, 2, centering, "+")
colnames(grid_colors) <- c("ug", "gr", "ri", "iz")

#--------------------------
# 3. GLM treinado nas cores (4D)
#--------------------------
glm_mod <- glm(class ~ ug + gr + ri + iz, family = binomial, data = df2)
grid$prob_glm <- predict(glm_mod, newdata = as.data.frame(grid_colors), type = "response")

#--------------------------
# 4. Neural Net (MLP nas cores)
#--------------------------
nn_mod <- nnet(
  x = as.matrix(df2[, c("ug", "gr", "ri", "iz")]),
  y = class.ind(df2$class),
  size = 50,
  softmax = TRUE,
  maxit = 300,
  decay = 1e-4,
  trace = FALSE
)
grid$prob_nn <- predict(nn_mod, newdata = grid_colors)[, 2]

#--------------------------
# 5. Random Forest (cores)
#--------------------------
rf_mod <- randomForest(class ~ ug + gr + ri + iz, data = df2, ntree = 500)
grid$prob_rf <- as.numeric(predict(rf_mod, newdata = as.data.frame(grid_colors), type = "prob")[, 2])

#--------------------------
# 6. Função de plotagem
#--------------------------
plot_model <- function(grid, prob_col, df, title) {
  ggplot() +
    geom_raster(data = grid, aes(x = x1, y = x2, fill = !!as.name(prob_col)), interpolate = TRUE) +
    geom_contour(data = grid, aes(x = x1, y = x2, z = !!as.name(prob_col)),
                 breaks = 0.5, color = "white", size = 0.8) +
    geom_point(data = df, aes(x = PC1, y = PC2, color = class, shape = class), size = 2) +
    scale_fill_viridis(name = "P(class=quasar)", option = "magma") +
    scale_color_manual(values = c("orange", "cyan3")) +
    coord_equal() +
    labs(title = title, x = "PC1", y = "PC2") +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none", panel.grid = element_blank()) +
    coord_cartesian(ylim = c(-2.5, 2.5),xlim=c(-5,5))
}

#--------------------------
# 7. Visualização final
#--------------------------
plot_glm <- plot_model(grid, "prob_glm", df2, "GLM (Cores → Projeção PCA)")
plot_nn  <- plot_model(grid, "prob_nn",  df2, "Neural Net (MLP, 50 unidades)")
plot_rf  <- plot_model(grid, "prob_rf",  df2, "Random Forest (500 árvores)")

print(plot_glm)
print(plot_nn)
print(plot_rf)
