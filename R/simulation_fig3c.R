box::purge_cache()
box::use(./generate_data[generate_data, generate_fixed_params])
box::use(./method[execute_proporsed_method, execute_base_method])

n_A_list <- c(1600, 3200, 6400, 12800)

library(future)
library(furrr)

plan(multisession)

result <- data.frame()
for (n_A in n_A_list) {
  print(sprintf("n_A = %f", n_A))
  set.seed(314)
  params <- generate_fixed_params(s = 2.4, M = 1)
  
  res <- future_map_dfr(1:10, function(i) {
    data <- generate_data(params, n_A = n_A, n_test = 1000)
    mse_proporsed <- execute_proporsed_method(data, params)
    mse_base <- execute_base_method(data)
    relative_mse <- mse_proporsed / mse_base
    data.frame(n_A = n_A, i = i, relative_mse = relative_mse)
  }, .options = furrr_options(seed = TRUE))
  
  result <- rbind(result, res)
}

saveRDS(result, "rds/fig3c.rds")

library(tidyverse)

df <- result |> 
  group_by(n_A) |>
  summarise(n = n(), mean = mean(relative_mse), sd = sd(relative_mse), se = sd / sqrt(n), 
            lower = mean - 1.96 * se, upper = mean + 1.96 * se, .groups = "drop")

ggplot(df, aes(n_A, mean)) +
  geom_line() +
  geom_pointrange(aes(ymin = lower, ymax = upper))
