box::purge_cache()
box::use(./generate_data[generate_data, generate_fixed_params])
box::use(./method[execute_proporsed_method, execute_base_method, 
                  execute_ImpNN_method])

M_list <- c(0.2, 0.6, 1, 2)

library(future)
library(furrr)

plan(multisession)

result <- data.frame()
for (M in M_list) {
  print(sprintf("M = %f", M))
  set.seed(314)
  params <- generate_fixed_params(s = 0.3, M = M)
  
  res <- future_map_dfr(1:10, function(i) {
    data <- generate_data(params, n_A = 6400, n_test = 1000)
    mse_proporsed <- execute_proporsed_method(data, params)
    mse_base <- execute_base_method(data)
    mse_impNN <- execute_ImpNN_method(data, params)
    relative_mse <- c(mse_proporsed, mse_impNN) / mse_base
    data.frame(M = M, i = i, method = c("BB-transfer", "Imp+NN"),
               relative_mse = relative_mse)
  }, .options = furrr_options(seed = TRUE))
  
  result <- rbind(result, res)
}

saveRDS(result, "rds/fig3b.rds")

library(tidyverse)

df <- result |> 
  group_by(M, method) |>
  summarise(n = n(), mean = mean(relative_mse), sd = sd(relative_mse), se = sd / sqrt(n), 
            lower = mean - 1.96 * se, upper = mean + 1.96 * se, .groups = "drop")

ggplot(df, aes(M, mean, color = method)) +
  geom_line() +
  geom_pointrange(aes(ymin = lower, ymax = upper))
