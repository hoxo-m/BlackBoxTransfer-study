box::purge_cache()
box::use(./generate_data[generate_data, generate_fixed_params])

# params <- generate_fixed_params(s = 0.03, M = 1)
params <- generate_fixed_params(s = 3.6, M = 1)

data <- generate_data(params, n_A = 6400, n_test = 1000)

box::use(./method[execute_proporsed_method, execute_base_method])

mse_proprsed <- execute_proporsed_method(data, params)
mse_base <- execute_base_method(data)

relative_mse <- mse_proprsed / mse_base
relative_mse
