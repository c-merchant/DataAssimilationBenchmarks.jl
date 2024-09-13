##############################################################################################
module TestVarAD
##############################################################################################
# imports and exports
using DataAssimilationBenchmarks
using DataAssimilationBenchmarks.XdVAR, DataAssimilationBenchmarks.ObsOperators
using DataAssimilationBenchmarks.L96, DataAssimilationBenchmarks.DeSolvers
using ForwardDiff, LinearAlgebra, Random, Distributions, LsqFit, StatsBase
##############################################################################################
"""
    test3DCost() 

    Tests the 3DVAR cost function for known behavior.
"""
function test3DCost()
    # initialization
    x = ones(40) * 0.5
    obs = zeros(40)
    x_bkg = ones(40)
    state_cov = 1.0I
    obs_cov = 1.0I
    params = Dict{String, Any}()
    H_obs = alternating_obs_operator

    cost = D3_var_cost(x, obs, x_bkg, state_cov, H_obs, obs_cov, params)

    if abs(cost - 10) < 0.001
        true
    else
        false
    end
end


##############################################################################################
"""
    test3DGrad() 

    Tests the gradient 3DVAR cost function for known behavior using ForwardDiff.
"""
function test3DGrad()
    # initialization
    obs = zeros(40)
    x_bkg = ones(40)
    state_cov = 1.0I
    obs_cov = 1.0I
    params = Dict{String, Any}()
    H_obs = alternating_obs_operator

    # wrapper function
    function wrap_cost(x)
        D3_var_cost(x, obs, x_bkg, state_cov, H_obs, obs_cov, params)
    end
    # input
    x = ones(40) * 0.5

    grad = ForwardDiff.gradient(wrap_cost, x)
    
    if norm(grad) < 0.001
        true
    else
        false
    end
end


##############################################################################################
"""
    test3DNewton() 

    Tests the Newton optimization of the 3DVAR cost function.
"""
function test3DNewton()
    # initialization
    obs = zeros(40)
    x_bkg = ones(40)
    state_cov = 1.0I
    obs_cov = 1.0I
    params = Dict{String, Any}()
    H_obs = alternating_obs_operator

    # perform Simple Newton optimization
    op = D3_var_NewtonOp(x_bkg, obs, state_cov, H_obs, obs_cov, params)
  
    if abs(sum(op - ones(40) * 0.5)) < 0.001
        true
    else
        false
    end
end


##############################################################################################
"""
    test3DNewtonNoise() 

    Tests the Newton optimization of the 3DVAR cost function with noise.
"""
function test3DNewtonNoise()
    # initialization
    Random.seed!(123)
    obs = rand(Normal(0, 1), 40)
    x_bkg = zeros(40)
    state_cov = 1.0I
    obs_cov = 1.0I
    params = Dict{String, Any}()
    H_obs = alternating_obs_operator

    # perform Simple Newton optimization
    op = D3_var_NewtonOp(x_bkg, obs, state_cov, H_obs, obs_cov, params)
    
    if abs(sum(op - obs * 0.5)) < 0.001
        true
    else
        false
    end
end


##############################################################################################
"""
    test3D()

    Tests the accuracy of the 3D-Var observation-analysis-forecast cycle.
"""
function test3DCycle()
    # 3D initialization
    x_t0 = ones(40)
    x_t1 = ones(40)
    delta = zeros(40)
    delta[1] = 0.01
    x_b1 = x_t0 + delta
    H_obs = alternating_obs_operator
    tanl = 0.1
    h = 0.01
    f_steps = Int64(tanl/h)
    state_cov = 1.0I
    obs_cov = 1.0I
    diffusion = 0.0
    dx_params = Dict{String, Array{Float64}}("F" => [8])
    params = Dict{String, Any}()
    kwargs = Dict{String, Any}(
            "h" => h,
            "diffusion" => diffusion,
            "dx_params" => dx_params,
            "dx_dt" => L96.dx_dt,
            "f_steps" => f_steps,
            )
    H_obs = alternating_obs_operator
    
    # evolve background and true states forward in time using RK4
    for i in 1:f_steps
        rk4_step!(x_t1, 1.0, kwargs)
        rk4_step!(x_b1, 1.0, kwargs)
    end

    # generate observations noise 
    Random.seed!(123)
    n = Normal()
    e_1 = rand(n, 40)
    y1 = H_obs(x_t1, length(x_t1), kwargs) + e_1

    # perform 3D-Var analysis using the Newton optimization operator
    x_a1 = D3_var_NewtonOp(x_b1, y1, state_cov, H_obs, obs_cov, kwargs)

    # calculate rmse
    rmse = sqrt(msd(x_a1, x_t1))

    if rmse - 0.4134 < 0.001
        true
    else
        false
    end

end


##############################################################################################
"""
    test4DLag1Cost() 

    Tests the 4DVAR Lag-1 cost function for known behavior.
"""
function test4DLag1Cost()
    # initialization
    x = ones(40) * 0.5
    obs = zeros(40)
    x_bkg = ones(40)
    state_cov = 1.0I
    obs_cov = 1.0I
    tanl = 0.1
    h = 0.01
    f_steps = Int64(tanl/h)
    diffusion = 0.0
    dx_params = Dict{String, Array{Float64}}("F" => [8])
    kwargs = Dict{String, Any}(
            "h" => h,
            "diffusion" => diffusion,
            "dx_params" => dx_params,
            "dx_dt" => L96.dx_dt,
            "f_steps" => f_steps,
            )
    H_obs = alternating_obs_operator

    cost = D4_var_lag1_cost(x, obs, x_bkg, state_cov, H_obs, obs_cov, kwargs)

    if abs(cost - 34.4623) < 0.001
        true
    else
        false
    end
end


##############################################################################################
"""
    test4DLag1Grad() 

    Tests the gradient 4DVAR Lag-1 cost function for known behavior using ForwardDiff.
"""
function test4DLag1Grad()
    # initialization
    obs = zeros(40)
    x_bkg = ones(40)
    state_cov = 1.0I
    obs_cov = 1.0I
    tanl = 0.1
    h = 0.01
    f_steps = Int64(tanl/h)
    diffusion = 0.0
    dx_params = Dict{String, Array{Float64}}("F" => [8])
    kwargs = Dict{String, Any}(
            "h" => h,
            "diffusion" => diffusion,
            "dx_params" => dx_params,
            "dx_dt" => L96.dx_dt,
            "f_steps" => f_steps,
            )
    H_obs = alternating_obs_operator

    # wrapper function
    function wrap_cost(x)
        D4_var_lag1_cost(x, obs, x_bkg, state_cov, H_obs, obs_cov, kwargs)
    end
    # input
    x = ones(40) * 0.5

    grad = ForwardDiff.gradient(wrap_cost, x)
    
    if norm(grad) - 3.974 < 0.001
        true
    else
        false
    end
end


##############################################################################################
"""
    test4DNewtonLag1() 

    Tests the Newton optimization of the 4DVar Lag-1 cost function.
"""
function test4DLag1Newton()
    # initialization
    obs = zeros(40)
    x_bkg = ones(40)
    state_cov = 1.0I
    obs_cov = 1.0I
    tanl = 0.1
    h = 0.01
    f_steps = Int64(tanl/h)
    diffusion = 0.0
    dx_params = Dict{String, Array{Float64}}("F" => [8])
    kwargs = Dict{String, Any}(
            "h" => h,
            "diffusion" => diffusion,
            "dx_params" => dx_params,
            "dx_dt" => L96.dx_dt,
            "f_steps" => f_steps,
            )
    H_obs = alternating_obs_operator

    # perform Simple Newton optimization
    op = D4_var_lag1_NewtonOp(x_bkg, obs, state_cov, H_obs, obs_cov, kwargs)
  
    if abs(sum(op - ones(40) * 0.17)) - 0.2586 < 0.001
        true
    else
        false
    end
end


##############################################################################################
"""
    test4DLag1NewtonNoise() 

    Tests the Newton optimization of the 4DVar Lag-1 cost function with noise.
"""
function test4DLag1NewtonNoise()
    # initialization
    Random.seed!(123)
    obs = rand(Normal(0, 1), 40)
    x_bkg = ones(40)
    state_cov = 1.0I
    obs_cov = 1.0I
    tanl = 0.1
    h = 0.01
    f_steps = Int64(tanl/h)
    diffusion = 0.0
    dx_params = Dict{String, Array{Float64}}("F" => [8])
    kwargs = Dict{String, Any}(
            "h" => h,
            "diffusion" => diffusion,
            "dx_params" => dx_params,
            "dx_dt" => L96.dx_dt,
            "f_steps" => f_steps,
            )
    H_obs = alternating_obs_operator

    # perform Simple Newton optimization
    op = D4_var_lag1_NewtonOp(x_bkg, obs, state_cov, H_obs, obs_cov, kwargs)
    
    if sum(op) - 5.461 < 0.001
        true
    else
        false
    end
end


##############################################################################################
"""
    test4DLag1Cycle()

    Tests the accuracy of the 4DVAR Lag-1 observation-analysis-forecast cycle.
"""
function test4DLag1Cycle()
    # initialization
    x_t0 = ones(40)
    x_t1 = ones(40)
    delta = zeros(40)
    delta[1] = 0.01
    x_b1 = x_t0 + delta
    H_obs = alternating_obs_operator
    tanl = 0.1
    h = 0.01
    f_steps = Int64(tanl/h)
    state_cov = 1.0I
    obs_cov = 1.0I
    diffusion = 0.0
    dx_params = Dict{String, Array{Float64}}("F" => [8])
    params = Dict{String, Any}()
    kwargs = Dict{String, Any}(
            "h" => h,
            "diffusion" => diffusion,
            "dx_params" => dx_params,
            "dx_dt" => L96.dx_dt,
            "f_steps" => f_steps,
            )
    H_obs = alternating_obs_operator

    # evolve true states forward in time using RK4
    for i in 1:f_steps
        rk4_step!(x_t1, 1.0, kwargs)
    end

    # generate noisy observations based on the true state
    Random.seed!(123)
    n = Normal()
    e_1 = rand(n, 40)
    y1 = H_obs(x_t1, length(x_t1), kwargs) + e_1


    # perform 4D-Var Lag-1 analysis using Newton optimization
    x_a = D4_var_lag1_NewtonOp(x_b1, y1, state_cov, H_obs, obs_cov, kwargs)

    # evolve the analysis state forward in time using RK4
    for i in 1:f_steps
        rk4_step!(x_a, 1.0, kwargs)
    end
    
    # calculate rmse
    rmse = sqrt(msd(x_a, x_t1))

    if rmse - 0.365 < 0.001
        true
    else
        false
    end
end


##############################################################################################
"""
    test4DLagLCost()

    Tests the 4DVAR Lag-L cost function for known behavior using dynamically generated observations.
"""
function test4DLagLCost()
    # initialization
    x_t0 = ones(40)
    x_t1 = ones(40)
    delta = zeros(40)
    delta[1] = 0.01
    x_b = x_t0 + delta
    H_obs = alternating_obs_operator
    tanl = 0.1
    h = 0.01
    f_steps = Int64(tanl / h)
    state_cov = Diagonal(ones(40))
    obs_cov = Diagonal(ones(40))
    diffusion = 0.0
    dx_params = Dict{String, Array{Float64}}("F" => [8])
    kwargs = Dict{String, Any}(
        "h" => h,
        "diffusion" => diffusion,
        "dx_params" => dx_params,
        "dx_dt" => L96.dx_dt,
        "f_steps" => f_steps,
    )
    H_obs = alternating_obs_operator
    lag = 4
    shift = 2
    
    # evolve and generate observations for 100 timesteps
    n_timesteps = 100
    if lag >= n_timesteps
        error("Lag value cannot exceed or equal the number of timesteps.")
    end

    obs = Matrix{Float64}(undef, 40, n_timesteps)
    x_current = copy(x_t1)
    Random.seed!(123)
    n = Normal()

    for t in 1:n_timesteps
        # evolve the state for one time step
        for i in 1:f_steps
            rk4_step!(x_current, 1.0, kwargs)
        end

        # generate noisy observation for the current timestep with decreasing noise variance
        noise_var = 1.0 / t
        noise = rand(n, 40) * sqrt(noise_var)
        obs[:, t] = H_obs(x_current, length(x_current), kwargs) + noise
    end

    # select the appropriate number of timesteps for lag
    y_stacked = obs[:, end-lag:end]

    # compute cost function
    cost = D4_var_lagL_cost(x_b, y_stacked, x_b, state_cov, H_obs, obs_cov, lag, shift, kwargs)

    if abs(cost - 1455.627) < 0.001
        true
    else
        false
    end
end


##############################################################################################
"""
    test4DGradLagL()

    Tests the gradient 4DVAR Lag-L cost function using dynamically generated observations.
"""
function test4DLagLGrad()
    # initialization
    x_t0 = ones(40)
    x_t1 = ones(40)
    delta = zeros(40)
    delta[1] = 0.01
    x_b = x_t0 + delta
    H_obs = alternating_obs_operator
    tanl = 0.1
    h = 0.01
    f_steps = Int64(tanl / h)
    state_cov = Diagonal(ones(40))
    obs_cov = Diagonal(ones(40))
    diffusion = 0.0
    dx_params = Dict{String, Array{Float64}}("F" => [8])
    kwargs = Dict{String, Any}(
        "h" => h,
        "diffusion" => diffusion,
        "dx_params" => dx_params,
        "dx_dt" => L96.dx_dt,
        "f_steps" => f_steps,
    )
    H_obs = alternating_obs_operator
    lag = 4
    shift = 2

    # evolve and generate observations for 100 timesteps
    n_timesteps = 100
    if lag >= n_timesteps
        error("Lag value cannot exceed or equal the number of timesteps.")
    end

    obs = Matrix{Float64}(undef, 40, n_timesteps)
    x_current = copy(x_t1)
    Random.seed!(123)
    n = Normal()

    for t in 1:n_timesteps
        # evolve the state for one time step
        for i in 1:f_steps
            rk4_step!(x_current, 1.0, kwargs)
        end

        # generate noisy observation for the current timestep with decreasing noise variance
        noise_var = 1.0 / t
        noise = rand(n, 40) * sqrt(noise_var)
        obs[:, t] = H_obs(x_current, length(x_current), kwargs) + noise
    end

    # select the appropriate number of timesteps for lag
    y_stacked = obs[:, end-lag:end]

    # wrapper function for ForwardDiff
    function wrap_cost(x)
        D4_var_lagL_cost(x, y_stacked, x_b, state_cov, H_obs, obs_cov, lag, shift, kwargs)
    end
    # input vector
    x = ones(40) * 0.5

    # compute gradient
    grad = ForwardDiff.gradient(wrap_cost, x)

    if abs(sum(grad + (ones(40) * 11.6538))) < 0.0021
        true
    else
        false
    end
end


##############################################################################################
"""
    test4DNewtonLagL()

    Tests the Newton optimization of the 4DVar Lag-L cost function with dynamically generated observations.
"""
function test4DLagLNewton()
    # initialization
    x_t0 = ones(40)
    x_t1 = ones(40)
    delta = zeros(40)
    delta[1] = 0.01
    x_b = x_t0 + delta
    H_obs = alternating_obs_operator
    tanl = 0.1
    h = 0.01
    f_steps = Int64(tanl / h)
    state_cov = Diagonal(ones(40))
    obs_cov = Diagonal(ones(40))
    diffusion = 0.0
    dx_params = Dict{String, Array{Float64}}("F" => [8])
    kwargs = Dict{String, Any}(
        "h" => h,
        "diffusion" => diffusion,
        "dx_params" => dx_params,
        "dx_dt" => L96.dx_dt,
        "f_steps" => f_steps,
    )
    H_obs = alternating_obs_operator
    lag = 4
    shift = 2

    # evolve and generate observations for 100 timesteps
    n_timesteps = 100
    if lag >= n_timesteps
        error("Lag value cannot exceed or equal the number of timesteps.")
    end

    obs = Matrix{Float64}(undef, 40, n_timesteps)
    x_current = copy(x_t1)
    Random.seed!(123)
    n = Normal()

    for t in 1:n_timesteps
        # evolve the state for one time step
        for i in 1:f_steps
            rk4_step!(x_current, 1.0, kwargs)
        end

        # generate noisy observation for the current timestep with decreasing noise variance
        noise_var = 1.0 / t
        noise = rand(n, 40) * sqrt(noise_var)
        obs[:, t] = H_obs(x_current, length(x_current), kwargs) + noise
    end

    # select the appropriate number of timesteps for lag
    y_stacked = obs[:, end-lag:end]

    # perform Newton optimization
    op = D4_var_lagL_NewtonOp(x_b, y_stacked, state_cov, H_obs, obs_cov, lag, shift, kwargs)

    if abs(sum(op - ones(40) * 5.1819)) < 0.001
        true
    else
        false
    end
end


##############################################################################################
"""
    test4DLagLCycle()

    Tests the accuracy of the 4DVAR Lag-L observation-analysis-forecast cycle with dynamically 
    generated observations to ensure larger lag produces lower RMSE.
"""
function test4DLagLCycle()
    # compare the rmse for the cases where Lag = 4, Shift = 2 and Lag = 2, Shift = 2
    if calc4DLagL(4,2) < calc4DLagL(2,2)
        true
    else
        false
    end
end


##############################################################################################
"""
    calc4DLagL(L::Int, S::Int)

    Function to perform 4DVAR Lag-L observation-analysis-forecast cycle with a data assimilation
    window shift of S and dynamically generated observations for test case.
"""
function calc4DLagL(L::Int, S::Int)
    # initialization
    x_t0 = ones(40)
    x_t1 = ones(40)
    delta = zeros(40)
    delta[1] = 0.01
    x_b = x_t0 + delta
    H_obs = alternating_obs_operator
    tanl = 0.1
    h = 0.01
    f_steps = Int64(tanl / h)
    state_cov = Diagonal(ones(40))
    obs_cov = Diagonal(ones(40))
    diffusion = 0.0
    dx_params = Dict{String, Array{Float64}}("F" => [8])
    kwargs = Dict{String, Any}(
        "h" => h,
        "diffusion" => diffusion,
        "dx_params" => dx_params,
        "dx_dt" => L96.dx_dt,
        "f_steps" => f_steps,
    )

    # evolve and generate observations for 100 timesteps
    n_timesteps = 100
    if L >= n_timesteps
        error("Lag value cannot exceed or equal the number of timesteps.")
    end

    obs = Matrix{Float64}(undef, 40, n_timesteps)
    x_current = copy(x_t1)
    Random.seed!(123)
    n = Normal()

    for t in 1:n_timesteps
        # evolve the state for one time step
        for i in 1:f_steps
            rk4_step!(x_current, 1.0, kwargs)
        end

        # generate noisy observation for the current timestep with decreasing noise variance
        noise_var = 1.0 / t
        noise = rand(n, 40) * sqrt(noise_var)
        obs[:, t] = H_obs(x_current, length(x_current), kwargs) + noise
    end

    # select the appropriate number of timesteps for lag
    y_stacked = obs[:, end-L:end]

    # perform the analysis with the specified lag and shift
    x_a = D4_var_lagL_NewtonOp(x_b, y_stacked, state_cov, H_obs, obs_cov, L, S, kwargs)

    # evolve the analysis state using RK4
    for i in 1:f_steps
        rk4_step!(x_a, 1.0, kwargs)
    end

    # calculate rmse
    rmse = sqrt(msd(x_a, x_current))

    return rmse
end


##############################################################################################
# end module

end
