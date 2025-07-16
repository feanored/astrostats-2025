library(tidyverse)   # dplyr, ggplot2 & afins
library(plotly)      # surface interativo (opcional)
library(plot3D)
library(scales)
library(nnet)
library(ggnewscale)
# ------------------------------------------------------------------
# 1. Dados filtrados (somente morfologia "E")
# ------------------------------------------------------------------
df0 <- read.csv("https://raw.githubusercontent.com/astrobayes/BMAD/refs/heads/master/data/Section_10p8/Seyfert.csv")
df <- df0 %>% filter(zoo == "E")

# ------------------------------------------------------------------
# 2. Paleta
# ------------------------------------------------------------------
pal <- c("orange2", "cyan2")

# ------------------------------------------------------------------
# 3. Grade de predição regular (300 x 300)
# ------------------------------------------------------------------
x1_seq <- seq(min(df$r_r200),  max(df$r_r200),  length.out = 300)
x2_seq <- seq(min(df$logM200), max(df$logM200), length.out = 300)
grid <- expand.grid(r_r200 = x1_seq, logM200 = x2_seq)

# ------------------------------------------------------------------
# 4. GLM (com termos quadráticos + interação)
# ------------------------------------------------------------------
glm_mod <- glm(bpt ~ logM200 + r_r200 + I(logM200^2) + I(r_r200^2) +
                 logM200:r_r200,
               data   = df,
               family = binomial(link = "logit"))

grid$prob_glm <- predict(glm_mod, newdata = grid, type = "response")

# ------------------------------------------------------------------
# 5. Neural Net
# ------------------------------------------------------------------
df_nn <- df %>%
  mutate(class = factor(bpt)) %>%
  select(r_r200, logM200, class)

nn_mod <- nnet(
  x = df_nn[, c("r_r200", "logM200")],
  y = class.ind(df_nn$class),
  size = 30,         # hidden units
  softmax = TRUE,
  maxit = 500,
  decay = 1e-4,
  trace = FALSE
)

grid$prob_nn <- predict(nn_mod, newdata = grid)[, 2]

# ------------------------------------------------------------------
# 6. Random Forest
# ------------------------------------------------------------------
rf_mod <- randomForest(class ~ r_r200 + logM200,
                       data = df_nn, ntree = 1000)

grid$prob_rf <- as.numeric(predict(rf_mod, newdata = grid, type = "prob")[, 2])

# ------------------------------------------------------------------
# 7. Função de plotagem
# ------------------------------------------------------------------
plot_model <- function(grid, prob_col, df, title) {
  ggplot() +
    geom_raster(data = grid, aes(x = r_r200, y = logM200, fill = !!as.name(prob_col)), interpolate = TRUE) +
    geom_contour(data = grid, aes(x = r_r200, y = logM200, z = !!as.name(prob_col)),
                 breaks = 0.5, colour = "white", linewidth = 0.8) +
    geom_point(data = df, aes(x = r_r200, y = logM200,
                              colour = factor(bpt), shape = factor(bpt)),
               size = 3) +
    scale_fill_continuous_sequential("Grays",name = "P(Seyfert)", limits = c(0, 1)) +
    scale_colour_manual(values = pal, name = "bpt") +
    scale_shape_manual(values = c(16, 17), name = "bpt") +
 #   coord_equal() +
    labs(title = title,
         x = expression(r/r[200]),
         y = expression(log[10]~M[200])) +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none", panel.grid = element_blank())
}

# ------------------------------------------------------------------
# 8. Plots
# ------------------------------------------------------------------
plot_glm <- plot_model(grid, "prob_glm", df, "GLM (Quadratic)")
plot_nn  <- plot_model(grid, "prob_nn",  df, "Neural Net (MLP, 15 hidden)")
plot_rf  <- plot_model(grid, "prob_rf",  df, "Random Forest (1000 trees)")

# Mostrar
print(plot_glm)
print(plot_nn)
print(plot_rf)
