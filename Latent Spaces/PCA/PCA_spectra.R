# -------------------------------------------------------------
# 0. Pacotes ---------------------------------------------------
library(tidyverse)      # dplyr + ggplot2
library(patchwork)      # empilhar plots

# -------------------------------------------------------------
# 1. Ler a matriz de espectros -------------------------------
#    (assumindo: espaço/tab, sem cabeçalho, 1000 colunas)
spec_mat <- as.matrix(read.table("sdss_spectra_1000.dat", header = FALSE))

# -------------------------------------------------------------
# 2. PCA  ------------------------------------------------------
#    (centra cada coluna; não escalona para σ=1, pois fluxos já
#     estão em unidades semelhantes)
pca <- prcomp(spec_mat, center = TRUE, scale. = FALSE)

# Variância explicada
var_explained <- (pca$sdev)^2 / sum(pca$sdev^2)
cumvar        <- cumsum(var_explained)

# -------------------------------------------------------------
# 3. Visualização da variância --------------------------------
df_var <- tibble(PC = 1:length(var_explained),
                 var = var_explained,
                 cumvar = cumvar)


ggplot(df_var, aes(PC, cumvar)) +
  geom_line() + geom_point() +
  labs(y = "Cumulative variance", x = "PC") +
  ylim(0, 1) + xlim(0,200)

# -------------------------------------------------------------
# 4. Reconstrução com k PCs -----------------------------------
k_choice <- 2  # << ajuste aqui

# Pontuações k-dimensionais
scores_k <- pca$x[, 1:k_choice]

# Cargas (eixos) k
loadings_k <- pca$rotation[, 1:k_choice]

# Reconstrução:  X̂ = scores × loadingsᵀ  + média
spec_recon <- scores_k %*% t(loadings_k)
spec_recon <- sweep(spec_recon, 2, pca$center, "+")  # devolve média

# -------------------------------------------------------------
# 5. Comparação visual ----------------------------------------
idx_show <- sample(1:nrow(spec_mat), 4)   # 4 espectros aleatórios

plots <- lapply(idx_show, function(i) {
  df <- tibble(
    lambda = 1:ncol(spec_mat),            # mude p/ wave_vec se tiver
    original = spec_mat[i, ],
    recon    = spec_recon[i, ]
  ) %>%
    pivot_longer(-lambda, names_to = "type", values_to = "flux")

  ggplot(df, aes(lambda, flux, colour = type)) +
    geom_line(alpha = 0.8) +
    scale_colour_manual(values = c(original = "black", recon = "red")) +
    labs(title = paste("Spectrum", i),
         x = "Pixel (or λ index)", y = "Flux") +
    theme_minimal()
})

wrap_plots(plots, ncol = 2)

# -------------------------------------------------------------
# 6. Salvar espectros reconstruídos (opcional) ----------------
write.table(spec_recon, sprintf("spec_recon_%dPCs.dat", k_choice),
            row.names = FALSE, col.names = FALSE)

