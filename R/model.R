box::use(torch[nn_dropout, nn_linear, nn_relu, nn_sequential])

make_MLP_model <- function(hidden_layers = 1, width = 8, dropout = 0.1) {
  layers <- list()
  in_features <- 11
  
  for (i in seq_len(hidden_layers)) {
    layers <- c(layers, list(
      nn_linear(in_features, width),
      nn_relu(),
      nn_dropout(p = dropout)
    ))
    in_features <- width
  }
  
  layers <- c(layers, list(nn_linear(in_features, 1)))
  
  do.call(nn_sequential, layers)
}
