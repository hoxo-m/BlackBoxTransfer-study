box::use(./execute_learning[execute_learning])
box::use(tibble[tibble])

#' @export
hypara_optim <- function(grid_hypara, dataloader_train, dataloader_val) {
  result <- tibble()
  for (i in seq_len(nrow(grid_hypara))) {
    hyper_parameters <- grid_hypara[i, ]
    # print(hyper_parameters)
    best <- execute_learning(dataloader_train, dataloader_val, hyper_parameters)
    res <- tibble(hyper_parameters, loss = best$val_mse, model = list(best$model))
    result <- rbind(result, res)
  }
  i_best <- which.min(result$loss)
  opt_model <- result$model[[i_best]]
  opt_model$eval()
  opt_model
}
