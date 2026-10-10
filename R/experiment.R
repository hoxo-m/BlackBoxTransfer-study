box::purge_cache()
box::use(./generate_data[generate_data, generate_fixed_params])
box::use(./method[execute_proporsed_method, execute_base_method])

s_list <- c(0.003, 0.6, 1.2, 3.6)

result <- data.frame()
for (s in s_list) {
  print(sprintf("s = %f", s))
  params <- generate_fixed_params(s = s, M = 1)
  
  for (i in 1:5) {
    data <- generate_data(params, n_A = 6400, n_test = 1000)
    
    mse_proprsed <- execute_proporsed_method(data, params)
    mse_base <- execute_base_method(data)
    
    relative_mse <- mse_proprsed / mse_base
    res <- data.frame(s = s, i = i, relative_mse)
    result <- rbind(result, res)
  }
}

library(tidyverse)

df <- result |> 
  group_by(s, i) |>
  summarise(n = n(), mean = mean(relative_mse), sd = sd(relative_mse), 
            se = sd / sqrt(n), lower = mean - se, upper = mean + se, .groups = "drop")

ggplot(df, aes(s, mean)) +
  geom_linerange(aes(ymin = lower, ymax = upper))
