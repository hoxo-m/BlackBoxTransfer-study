box::use(torch[nnf_mse_loss, with_no_grad])
box::use(coro[loop])
box::use(R6[R6Class])

#' @export
Learner <- R6Class(
  "Leaner",
  private = list(
    model = NULL,
    optimizer = NULL,
    dataloader_train = NULL,
    dataloader_val = NULL,
    n_train = 0,
    n_val = 0,
    count_samples = function(dataloader) {
      n <- 0
      loop(for (batch in dataloader) {
        x <- batch[[1]]
        n <- n + x$size(1)
      })
      n
    }
  ),
  public = list(
    initialize = function(model, optimizer, 
                          dataloader_train, dataloader_val) {
      private$model <- model
      private$optimizer <- optimizer
      private$dataloader_train <- dataloader_train
      private$dataloader_val <- dataloader_val
      private$n_train <- private$count_samples(dataloader_train)
      private$n_val <- private$count_samples(dataloader_val)
    },
    train = function() {
      private$model$train()
      
      total_loss <- 0
      
      loop(for (batch in private$dataloader_train) {
        x <- batch[[1]]
        y <- batch[[2]]
        
        private$optimizer$zero_grad()
        
        pred <- private$model(x)
        loss <- nnf_mse_loss(pred, y)
        
        loss$backward()
        private$optimizer$step()
        
        total_loss <- total_loss + loss$item() * x$size(1)
      })
      
      train_loss <- total_loss / private$n_train
      train_loss
    }, 
    validate = function() {
      private$model$eval()
      
      total_loss <- 0
      with_no_grad({
        loop(for (batch in private$dataloader_val) {
          x <- batch[[1]]
          y <- batch[[2]]
          
          pred <- private$model(x)
          loss <- nnf_mse_loss(pred, y)
          
          total_loss <- total_loss + loss$item() * x$size(1)
        })
      })
      
      val_loss <- total_loss / private$n_val
      val_loss
    }
  )
)
