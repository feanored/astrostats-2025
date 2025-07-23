# ================================================================
# 0. Packages -----------------------------------------------------
# ================================================================
# install.packages(c("tidyverse", "viridis", "nnet", "randomForest"))  # first time only
library(tidyverse)      # dplyr, ggplot2, …
library(viridis)        # colour scales
library(nnet)           # MLP
library(randomForest)   # Random Forest
require(colorspace)

# ================================================================
# 1. Data & basic pre-processing ---------------------------------
# ================================================================
df <- read_csv(
  "https://raw.githubusercontent.com/RafaelSdeSouza/astrostats-2025/refs/heads/main/Data/Exercises/rrlyrae_combined.csv"
)

df <- df %>% mutate(class = factor(class))   # ensure ‘class’ is a factor

# ================================================================
# 2. Cartesian grid in (ug, gr) space ----------------------------
# ================================================================
ug_seq <- seq(min(df$ug), max(df$ug), length.out = 300)
gr_seq <- seq(min(df$gr), max(df$gr), length.out = 300)
grid   <- expand.grid(ug = ug_seq, gr = gr_seq)

# ================================================================
# 3. GLM  (logistic, 2-D) ----------------------------------------
# ================================================================
glm_mod        <- glm(class ~ ug + gr + I(ug^2) + I(gr^2), family = binomial, data = df)
grid$prob_glm  <- predict(glm_mod, newdata = grid, type = "response")

# ================================================================
# 4. Neural Net  (MLP, 2-D) --------------------------------------
# ================================================================
nn_mod <- nnet(
  x       = as.matrix(df[, c("ug", "gr")]),
  y       = class.ind(df$class),   # one-hot targets
  size    = 25,                    # hidden units
  softmax = TRUE,                  # classification mode
  maxit   = 300,
  decay   = 1e-4,
  trace   = FALSE
)
grid$prob_nn <- predict(nn_mod, newdata = as.matrix(grid))[, 2]

# ================================================================
# 5. Random Forest  (2-D) ----------------------------------------
# ================================================================
rf_mod        <- randomForest(class ~ ug + gr, data = df, ntree = 1000)
grid$prob_rf  <- as.numeric(predict(rf_mod, newdata = grid, type = "prob")[, 2])

# ================================================================
# 6. Plot helper --------------------------------------------------
# ================================================================
plot_model <- function(grid, prob_col, df,title) {
  ggplot() +
    geom_raster(data = grid,
                aes(x = ug, y = gr, fill = !!as.name(prob_col)),
                interpolate = TRUE) +
 #   geom_contour(data = grid,
#                 aes(x = ug, y = gr, z = !!as.name(prob_col)),
#                 breaks = 0.25, colour = "red2", linewidth = 0.8) +
    geom_point(data = df,
               aes(x = ug, y = gr, colour = class, shape = class,size=class)) +
    scale_size_manual(values=c(0.2,1))+
    scale_fill_continuous_sequential("OrRd") +
    scale_colour_manual(values = c("orange", "red2")) +
    labs(title=title, x = "u − g", y = "g − r") +
    theme_minimal(base_size = 14) +
    theme(panel.grid = element_blank(),
          legend.position = "none")
}

# ================================================================
# 7. Visualisation -----------------------------------------------
# ================================================================
plot_glm <- plot_model(grid, "prob_glm", df, "GLM")
plot_nn  <- plot_model(grid, "prob_nn",  df, "Neural Net")
plot_rf  <- plot_model(grid, "prob_rf",  df, "Random Forest")

print(plot_glm)
print(plot_nn)
print(plot_rf)

