box::use(R6[R6Class])

History <- R6Class(
  "History",
  private = list(
    history = data.frame(
      epoch = integer(),
      loss_train = numeric(),
      loss_val = numeric()
    ),
    best_loss_validation = Inf,
    best_epoch = NA_integer_,
    best_state = NULL,
    epochs_no_improvement = 0,
    patience = Inf,
    model = NULL,
    clone_model_state = function() {
      lapply(private$model$state_dict(), function(x) x$detach()$clone())
    }
  ), 
  public = list(
    initialize = function(model, patience = Inf) {
      try({
        patience <- as.integer(patience)
        if (patience >= 1L) {
          private$patience <- patience
        }
      })
      private$model <- model
    },
    update = function(epoch, loss_train, loss_validation) {
      if (!is.finite(loss_validation)) {
        stop("Validation loss is not finite.")
      }
      
      i <- nrow(private$history) + 1
      private$history[i, ] <- list(epoch, loss_train, loss_validation)
      
      if (loss_validation < private$best_loss_validation) {
        private$best_loss_validation <- loss_validation
        private$best_epoch <- epoch
        private$epochs_no_improvement <- 0L
        
        # Copy parameter tensors
        private$best_state <- private$clone_model_state()
          
      } else {
        private$epochs_no_improvement <- private$epochs_no_improvement + 1L
      }
    },
    should_stop = function() {
      private$epochs_no_improvement >= private$patience
    },
    get_history = function() {
      private$history
    },
    restore_best = function() {
      if (is.null(private$best_state)) {
        stop("No best model has been saved.")
      }
      
      private$model$load_state_dict(private$best_state)
      private$model$eval()
      list(
        epoch = private$best_epoch,
        loss_validation = private$best_loss_validation,
        state = private$best_state,
        model = private$model
      )
    }
  )
)
