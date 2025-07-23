compare_activations <- function(model, batch, layer_name = "conv1", n_filters = 12,
                                index_round = NULL, index_barred = NULL) {
  activation_model <- keras_model(
    inputs = model$input,
    outputs = get_layer(model, layer_name)$output
  )

  labels <- batch[[2]]
  imgs <- batch[[1]]

  if (is.null(index_round)) index_round <- which(labels == 0)[1]
  if (is.null(index_barred)) index_barred <- which(labels == 1)[1]

  img_round <- array(imgs[index_round,,,], dim = c(1, dim(imgs)[2:4]))
  img_barred <- array(imgs[index_barred,,,], dim = c(1, dim(imgs)[2:4]))

  act_round <- predict(activation_model, img_round)
  act_barred <- predict(activation_model, img_barred)

  all_df <- data.frame()

  for (i in 1:n_filters) {
    df_round <- as.data.frame(as.table(act_round[1,,,i]))
    colnames(df_round) <- c("x", "y", "value")
    df_round$galaxy_type <- "Round Smooth"
    df_round$filter_id <- paste0("Filter ", i)

    df_barred <- as.data.frame(as.table(act_barred[1,,,i]))
    colnames(df_barred) <- c("x", "y", "value")
    df_barred$galaxy_type <- "Barred Spiral"
    df_barred$filter_id <- paste0("Filter ", i)

    all_df <- bind_rows(all_df, df_round, df_barred)
  }

  # 🔁 Define explicit order for galaxy types (top: Barred Spiral, bottom: Round Smooth)
  all_df$galaxy_type <- factor(all_df$galaxy_type, levels = c("Barred Spiral", "Round Smooth"))

  ggplot(all_df, aes(x = x, y = y, fill = value)) +
    geom_raster() +
    scale_fill_viridis_c() +
    coord_fixed() +
    facet_grid(rows = vars(galaxy_type), cols = vars(filter_id)) +
    theme_void(base_size = 12) +
    theme(
      strip.text = element_text(size = 12, face = "bold"),
      strip.background = element_blank(),
      plot.title = element_text(size = 16, face = "bold"),
      legend.position = "none",
      strip.text.y = element_text(angle = 90)
    ) +
    labs(title = paste("Activations from", layer_name))
}
