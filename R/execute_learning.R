box::use(torch[optim_adam])
box::use(./model[make_MLP_model])
box::use(./learner[Learner])
box::use(./history[History])

execute_learning <- function(dataloader_train, dataloader_val, hyparams) {
  model <- make_MLP_model(hidden_layers = hyparams$hidden_layers, 
                          width = hyparams$width, dropout = hyparams$dropout)
  optimizer <- optim_adam(model$parameters, lr = 3e-3, weight_decay = 1e-5)
  
  learner <- Learner$new(model, optimizer, dataloader_train, dataloader_val)
  
  patience <- 20
  history <- History$new(model, patience)
  
  max_epochs <- 200
  for (epoch in seq_len(max_epochs)) {
    train_mse <- learner$train()
    val_mse <- learner$validate()
    history$update(epoch, train_mse, val_mse)
    
    # --- Early Stopping ---
    if (history$should_stop()) {
      break
    }
  }
  
  # Restore best model
  best <- history$restore_best()
  
  list(
    model = best$model,
    val_mse = best$loss_validation,
    best_epoch = best$epoch,
    history = history$get_history()
  )
}
