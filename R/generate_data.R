box::use(stats[rnorm, rt, runif, sd])
box::use(mvtnorm[rmvt])

compute_L1_norm <- function(x) {
  sum(abs(x))
}

compute_L2_norm <- function(x) {
  sqrt(sum(x^2))
}

#' @export
generate_fixed_params <- function(s, M) {
  B <- matrix(nrow = 11, ncol = 10)
  for (k in 1:10) {
    b <- runif(11)
    norm_L1 <- compute_L1_norm(b) 
    B[, k] <- b / norm_L1
  }
  omega <- t(rmvt(n = 5, sigma = diag(10), df = 2))
  phi <- runif(5, min = 0, max = 2 * pi)
  f <- function(z) {
    total <- 0
    for (k in 1:5) {
      total <- total + cos(10 * sum(omega[, k] * z) + phi[k])
    }
    sqrt(2/5) * 1/5 * total
  }
  beta <- matrix(nrow = 11, ncol = 2)
  for (i in 1:2) {
    beta_i <- rnorm(11)
    norm_L2 <- compute_L2_norm(beta_i)
    beta[, i] <- beta_i / norm_L2
  }
  delta <- function(x) {
    M * tanh(
      sin(pi * t(x) %*% beta[, 1]) * 
        abs(2 * (x %*% beta[, 2] - 0.5) / sqrt(11))^s
    )
  }
  list(B = B, f = f, delta = delta)
}

generate_D_A <- function(n, params) {
  X <- matrix(nrow = n, ncol = 11)
  Z <- matrix(nrow = n, ncol = 10)
  for (i in seq_len(n)) {
    X[i,] <- matrix(runif(11))
    v <- t(matrix(runif(10)))
    Z[i,] <- 0.9 * t(X[i,]) %*% params$B + 0.1 * v
  }
  list(X = X, Z = Z)
}

generate_D_T <- function(n, params, sigma, n_rep) {
  X <- matrix(nrow = n, ncol = 11)
  h <- double(n)
  delta <- double(n)
  for (i in seq_len(n)) {
    X[i,] <- matrix(runif(11))
    f <- double(n_rep)
    tXB <- 0.9 * t(X[i,]) %*% params$B
    for (j in seq_len(n_rep)) {
      v <- t(matrix(runif(10, min = 0, max = 0.1)))
      z <- tXB + v
      f[j] <- params$f(z)
    }
    h[i] <- mean(f)
    delta[i] <- params$delta(X[i,])
  }
  Y <- h + delta + rnorm(n, 0, sd = sigma)
  list(X = X, Y = Y)
}

estimate_sigma <- function(delta, n = 10000) {
  X <- matrix(runif(n * 11), ncol = 11)
  sd(apply(X, 1, delta))
}

#' @export
generate_data <- function(params, n_A = 6400, 
                          n_T = 400, n_test = 10000, n_rep = 100) {

  sigma <- estimate_sigma(params$delta)
  
  D_A <- generate_D_A(n_A, params)
  D_T <- generate_D_T(n_T, params, sigma, n_rep)
  D_test <- generate_D_T(n_test, params, sigma, n_rep)
  list(D_A = D_A, D_T = D_T, D_test = D_test)
}
