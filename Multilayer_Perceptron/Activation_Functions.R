library(ggplot2)
library(patchwork)

# Dados base
x <- seq(-5, 5, length.out = 200)
#z <- ifelse(x > 2, 2*x + 1, -.5*x + 1)  # saída da regressão linear
z <- x^3 - 2*x^2 -x


# Ativações
relu    <- pmax(0, z)
sigmoid <- 1 / (1 + exp(-z))
tanhx   <- tanh(z)

# Define função de ativação genérica
activation_plot <- function(name, z_vals, act_vals, final_vals) {
  df1 <- data.frame(x, z)
  df2 <- data.frame(z = seq(-6, 6, length.out = 200))
  df2$activation <- switch(name,
                           "ReLU"    = pmax(0, df2$z),
                           "Sigmoid" = 1 / (1 + exp(-df2$z)),
                           "Tanh"    = tanh(df2$z)
  )
  df3 <- data.frame(x, y = final_vals)

  # Painel 1: saída linear
  p1 <- ggplot(df1, aes(x, z)) +
    geom_line(size = 1.2) +
    labs(title = "Linear: z = w₀ + w₁x", y = "z", x = "x") +
    theme_minimal(base_size = 13)

  # Painel 2: ativação
  p2 <- ggplot(df2, aes(z, activation)) +
    geom_line(color = "darkorange", size = 1.2) +
    labs(title = paste0("Função ", name), x = "z", y = paste0(name, "(z)")) +
    theme_minimal(base_size = 13)

  # Painel 3: ativação composta
  p3 <- ggplot(df3, aes(x, y)) +
    geom_line(color = "steelblue", size = 1.2) +
    labs(title = paste0("Saída: ", name, "(z)"), y = "y", x = "x") +
    theme_minimal(base_size = 13)

  return(p1 | p2 | p3)
}

# Mostrar para cada ativação
activation_plot("ReLU",    z, pmax(0, z), pmax(0, z))
activation_plot("Sigmoid", z, sigmoid,   sigmoid)
activation_plot("Tanh",    z, tanhx,     tanhx)

