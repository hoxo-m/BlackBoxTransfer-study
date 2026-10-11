box::use(torch[tensor_dataset, torch_float, torch_tensor])
box::use(torch[dataloader, dataset_subset])
box::use(./hyper_parameter_optimization[hypara_optim])
box::use(torch[with_no_grad, nnf_mse_loss])
box::use(./learner[Learner])
box::use(./execute_learning[execute_learning])
box::use(stats[lm])

#' @export
execute_proporsed_method <- function(data, params) {

  # Estimate h() ------------------------------------------------------------
  x <- torch_tensor(data$D_A$X, dtype = torch_float())
  y_value <- apply(data$D_A$Z, 1, params$f)
  y <- torch_tensor(matrix(y_value, ncol = 1), dtype = torch_float())
  
  ds <- tensor_dataset(x, y)
  
  n <- nrow(data$D_A$X)
  n_train <- floor(n * 0.8)
  n_val <- n - n_train
  inds <- sample.int(n)
  
  dataset_train <- dataset_subset(ds, inds[1:n_train])
  dataset_val <- dataset_subset(ds, inds[(n_train + 1):n])
  
  dataloader_train <- dataloader(dataset_train, batch_size = 64, shuffle = TRUE)
  dataloader_val <- dataloader(dataset_val, batch_size = 64, shuffle = FALSE)
  
  grid_hypara <- expand.grid(hidden_layers = c(1, 2), width = c(8, 16), 
                             dropout = c(0.1, 0.2))
  model_h_opt <- hypara_optim(grid_hypara, dataloader_train, dataloader_val)

  # Estimate delta() --------------------------------------------------------
  x <- torch_tensor(data$D_T$X, dtype = torch_float())
  y_value <- data$D_T$Y
  y <- torch_tensor(matrix(y_value, ncol = 1), dtype = torch_float())
  target <- with_no_grad(y - model_h_opt(x))
  
  ds <- tensor_dataset(x, target)
  
  n <- nrow(data$D_T$X)
  n_train <- floor(n * 0.8)
  n_val <- n - n_train
  inds <- sample.int(n)
  
  dataset_train <- dataset_subset(ds, inds[1:n_train])
  dataset_val <- dataset_subset(ds, inds[(n_train + 1):n])
  
  dataloader_train <- dataloader(dataset_train, batch_size = 64, shuffle = TRUE)
  dataloader_val <- dataloader(dataset_val, batch_size = 64, shuffle = FALSE)
  
  grid_hypara <- expand.grid(hidden_layers = c(2, 4), width = c(64, 128), 
                             dropout = c(0.1, 0.2))
  
  model_delta_opt <- hypara_optim(grid_hypara, dataloader_train, dataloader_val)

  # Estimate lambda ---------------------------------------------------------
  ds0 <- tensor_dataset(x, y)
  dataset_val0 <- dataset_subset(ds0, inds[(n_train + 1):n])
  dataloader_val0 <- dataloader(dataset_val0, batch_size = 64, shuffle = FALSE)
  learner_lambda0 <- Learner$new(model_h_opt, NULL, NULL, dataloader_val0)
  learner_lambda1 <- Learner$new(model_delta_opt, NULL, NULL, dataloader_val)

  df_loss <- data.frame(
    lambda = 0:1,
    loss = c(learner_lambda0$validate(), learner_lambda1$validate())
  )
  
  lambda_opt <- df_loss$lambda[which.min(df_loss$loss)]
  
  # Compute MSE -------------------------------------------------------------
  g <- if (lambda_opt == 0) {
    model_h_opt 
  } else {
    function(x) {
      model_h_opt(x) + model_delta_opt(x)
    }
  }
  
  x <- torch_tensor(data$D_test$X, dtype = torch_float())
  y_value <- data$D_test$Y
  y <- torch_tensor(matrix(y_value, ncol = 1), dtype = torch_float())
  
  mse <- with_no_grad(nnf_mse_loss(g(x), y)$item())
  mse
}

#' @export
execute_base_method <- function(data) {

  # Estimate base model -----------------------------------------------------
  x <- torch_tensor(data$D_T$X, dtype = torch_float())
  y_value <- data$D_T$Y
  y <- torch_tensor(matrix(y_value, ncol = 1), dtype = torch_float())
  
  ds <- tensor_dataset(x, y)
  
  n <- nrow(data$D_T$X)
  n_train <- floor(n * 0.8)
  n_val <- n - n_train
  inds <- sample.int(n)
  
  dataset_train <- dataset_subset(ds, inds[1:n_train])
  dataset_val <- dataset_subset(ds, inds[(n_train + 1):n])
  
  dataloader_train <- dataloader(dataset_train, batch_size = 64, shuffle = TRUE)
  dataloader_val <- dataloader(dataset_val, batch_size = 64, shuffle = FALSE)
  
  grid_hypara <- expand.grid(hidden_layers = c(2, 4), width = c(64, 128), 
                             dropout = c(0.1, 0.2))
  
  model_base_opt <- hypara_optim(grid_hypara, dataloader_train, dataloader_val)
  
  # Compute MSE -------------------------------------------------------------
  x <- torch_tensor(data$D_test$X, dtype = torch_float())
  y_value <- data$D_test$Y
  y <- torch_tensor(matrix(y_value, ncol = 1), dtype = torch_float())
  
  g <- model_base_opt
  
  mse <- with_no_grad(nnf_mse_loss(g(x), y)$item())
  mse
}

#' @export
execute_ImpNN_method <- function(data, params) {
  
  # Estimate Linear Model ---------------------------------------------------
  x <- data$D_A$X
  z <- data$D_A$Z
  
  fit <- lm(z ~ x)
  
  m_hat <- function(X) {
    cbind(1, X) %*% fit$coefficients
  }

  # Estimate delta() --------------------------------------------------------
  x_value <- data$D_T$X
  x <- torch_tensor(x_value, dtype = torch_float())
  h_hat_value <- apply(m_hat(x_value), 1, params$f)
  h_hat <- torch_tensor(matrix(h_hat_value, ncol = 1), dtype = torch_float())
  y_value <- data$D_T$Y
  y <- torch_tensor(matrix(y_value, ncol = 1), dtype = torch_float())
  target <- with_no_grad(y - h_hat)
  
  ds <- tensor_dataset(x, target)
  
  n <- nrow(data$D_T$X)
  n_train <- floor(n * 0.8)
  n_val <- n - n_train
  inds <- sample.int(n)
  
  dataset_train <- dataset_subset(ds, inds[1:n_train])
  dataset_val <- dataset_subset(ds, inds[(n_train + 1):n])
  
  dataloader_train <- dataloader(dataset_train, batch_size = 64, shuffle = TRUE)
  dataloader_val <- dataloader(dataset_val, batch_size = 64, shuffle = FALSE)
  
  grid_hypara <- expand.grid(hidden_layers = c(2, 4), width = c(64, 128), 
                             dropout = c(0.1, 0.2))
  
  model_delta_opt <- hypara_optim(grid_hypara, dataloader_train, dataloader_val)
  
  # Estimate lambda ---------------------------------------------------------
  inds_val <- inds[(n_train + 1):n]
  x_val <- x_value[inds_val, ]
  y_hat_value <- apply(m_hat(x_val), 1, params$f)
  y_hat_val <- torch_tensor(matrix(y_hat_value, ncol = 1), dtype = torch_float())
  y_val <- torch_tensor(matrix(y_value[inds_val], ncol = 1), dtype = torch_float())
  mse0 <- with_no_grad(nnf_mse_loss(y_hat_val, y_val)$item())
  
  learner_lambda1 <- Learner$new(model_delta_opt, NULL, NULL, dataloader_val)
  
  df_loss <- data.frame(
    lambda = 0:1,
    loss = c(mse0, learner_lambda1$validate())
  )
  
  lambda_opt <- df_loss$lambda[which.min(df_loss$loss)]
  
  # Compute MSE -------------------------------------------------------------
  g <- if (lambda_opt == 0) {
    function(x_value, x) {
      h_hat_value <- apply(m_hat(x_value), 1, params$f)
      torch_tensor(matrix(h_hat_value, ncol = 1), dtype = torch_float())
    }
  } else {
    function(x_value, x) {
      h_hat_value <- apply(m_hat(x_value), 1, params$f)
      torch_tensor(matrix(h_hat_value, ncol = 1), dtype = torch_float()) +
      model_delta_opt(x)
    }
  }
  
  x_value <- data$D_test$X
  x <- torch_tensor(x_value, dtype = torch_float())
  y_value <- data$D_test$Y
  y <- torch_tensor(matrix(y_value, ncol = 1), dtype = torch_float())
  
  mse <- with_no_grad(nnf_mse_loss(g(x_value, x), y)$item())
  mse
}
