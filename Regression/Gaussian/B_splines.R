library(splines)
library(ggplot2)
library(tidyr)
library(dplyr)

# Step 1: x range
x <- seq(0, 4, length.out = 200)

# Step 2: knot vector
knots <- c(0, 0, 0, 1, 2, 3, 4, 4, 4)  # degree 2 → 3 repeated knots at ends
degree <- 2
order <- degree + 1

# Step 3: B-spline basis matrix
B <- splineDesign(knots = knots, x = x, ord = order)

# Check number of basis functions:
n_basis <- ncol(B)  # should be 6 in this case

# Step 4: coefficients (must match number of basis functions!)
coeffs <- c(0, 1, 2, 1, 0, -1)

# Step 5: compute spline
spline_y <- B %*% coeffs

# Step 6: plot basis + spline
df_basis <- as.data.frame(B)
colnames(df_basis) <- paste0("B", 1:ncol(df_basis))
df_basis$x <- x
df_long <- pivot_longer(df_basis, -x, names_to = "Basis", values_to = "Value")

df_curve <- data.frame(x = x, y = as.vector(spline_y))

ggplot() +
  geom_line(data = df_long, aes(x, Value, color = Basis), alpha = 0.4) +
  geom_line(data = df_curve, aes(x, y), size = 1.5, color = "black") +
  labs(title = "B-spline Curve as Combination of Basis Functions",
       y = "Spline Value") +
  theme_minimal()

