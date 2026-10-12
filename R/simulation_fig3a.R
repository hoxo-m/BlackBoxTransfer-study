box::purge_cache()
box::use(./generate_data[generate_data, generate_fixed_params])
box::use(./method[execute_proporsed_method, execute_base_method, 
                  execute_ImpNN_method])

s_list <- c(0.03, 0.6, 1.2, 3.6)

library(future)
library(furrr)

plan(multisession)

result <- data.frame()
for (s in s_list) {
  print(sprintf("s = %f", s))
  set.seed(314)
  params <- generate_fixed_params(s = s, M = 1)

  res <- future_map_dfr(1:10, function(i) {
    data <- generate_data(params, n_A = 6400, n_test = 1000)
    mse_proporsed <- execute_proporsed_method(data, params)
    mse_base <- execute_base_method(data)
    mse_impNN <- execute_ImpNN_method(data, params)
    relative_mse <- c(mse_proporsed, mse_impNN) / mse_base
    data.frame(s = s, i = i, method = c("BB-transfer", "Imp+NN"),
               relative_mse = relative_mse)
  }, .options = furrr_options(seed = TRUE))
  
  result <- rbind(result, res)
}

saveRDS(result, "rds/fig3a.rds")

library(tidyverse)

df <- result |> 
  group_by(s, method) |>
  summarise(n = n(), mean = mean(relative_mse), sd = sd(relative_mse), se = sd / sqrt(n), 
            lower = mean - 1.96 * se, upper = mean + 1.96 * se, .groups = "drop")

ggplot(df, aes(s, mean, color = method)) +
  geom_line() +
  geom_pointrange(aes(ymin = lower, ymax = upper))
