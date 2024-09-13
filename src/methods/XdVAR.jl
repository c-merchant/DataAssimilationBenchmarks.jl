##############################################################################################
module XdVAR
##############################################################################################
# imports and exports
using LinearAlgebra, ForwardDiff
using ..DataAssimilationBenchmarks, DataAssimilationBenchmarks.DeSolvers
export D3_var_cost, D3_var_grad, D3_var_hessian, D3_var_NewtonOp, D4_var_lag1_cost, D4_var_lag1_grad
export D4_var_lag1_cost, D4_var_lag1_grad, D4_var_lag1_hessian, D4_var_lag1_NewtonOp 
export D4_var_lagL_cost, D4_var_lagL_grad, D4_var_lagL_hessian, D4_var_lagL_NewtonOp
##############################################################################################
# Main methods
##############################################################################################
"""
    D3_var_cost(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
        H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real

Computes the cost of the three-dimensional variational analysis increment from an initial state 
proposal with a static background covariance

`x` is a free argument used to evaluate the cost of the given state proposal versus other 
proposal states, `obs` is to the observation vector, `x_bkg` is the initial state proposal 
vector, `state_cov` is the background error covariance matrix, `H_obs` is a model mapping
operator for observations, and `obs_cov` is the observation error covariance matrix.
`kwargs` refers to any additional arguments needed for the operation computation.

```
return  0.5*back_component + 0.5*obs_component
```
"""
function D3_var_cost(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
                     H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real

    # initializations
    obs_dim = length(obs)

    # obs operator
    H = H_obs(x, obs_dim, kwargs)

    # background discrepancy
    δ_b = x - x_bkg

    # observation discrepancy
    δ_o = obs - H

    # cost function
    back_component = dot(δ_b, inv(state_cov) * δ_b)
    obs_component = dot(δ_o, inv(obs_cov) * δ_o)

    0.5*back_component + 0.5*obs_component
end


##############################################################################################
"""
    D3_var_grad(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
        H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Float64

Computes the gradient of the three-dimensional variational analysis increment from an initial 
state proposal with a static background covariance using a wrapper function for automatic 
differentiation

`x` is a free argument used to evaluate the cost of the given state proposal versus other 
proposal states, `obs` is to the observation vector, `x_bkg` is the initial state proposal 
vector, `state_cov` is the background error covariance matrix, `H_obs` is a model mapping
operator  for observations, and `obs_cov` is the observation error covariance matrix.
`kwargs` refers to any additional arguments needed for the operation computation.

`wrap_cost` is a function that allows differentiation with respect to the free argument `x`
while treating all other hyperparameters of the cost function as constant.

```
return  ForwardDiff.gradient(wrap_cost, x)
```
"""
function D3_var_grad(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
                     H_obs::Function, obs_cov::CovM(T),
                     kwargs::StepKwargs) where T <: Float64

    function wrap_cost(x::VecA(T)) where T <: Real
        D3_var_cost(x, obs, x_bkg, state_cov, H_obs, obs_cov, kwargs)
    end

    ForwardDiff.gradient(wrap_cost, x)
end


##############################################################################################
"""
    D3_var_hessian(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
        H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Float64 

Computes the Hessian of the three-dimensional variational analysis increment from an initial 
state proposal with a static background covariance using a wrapper function for automatic 
differentiation

`x` is a free argument used to evaluate the cost of the given state proposal versus other 
proposal states, `obs` is to the observation vector, `x_bkg` is the initial state proposal 
vector, `state_cov` is the background error covariance matrix, `H_obs` is a model mapping 
operator for observations, and `obs_cov` is the observation error covariance matrix.
`kwargs` refers to any additional arguments needed for the operation computation.

`wrap_cost` is a function that allows differentiation with respect to the free argument `x`
while treating all other hyperparameters of the cost function as constant.

```
return  ForwardDiff.hessian(wrap_cost, x)
```
"""
function D3_var_hessian(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
                        H_obs::Function, obs_cov::CovM(T),
                        kwargs::StepKwargs) where T <: Float64 

    function wrap_cost(x::VecA(T)) where T <: Real
        D3_var_cost(x, obs, x_bkg, state_cov, H_obs, obs_cov, kwargs)
    end

    ForwardDiff.hessian(wrap_cost, x)
end


##############################################################################################
"""
    D3_var_NewtonOp(x_bkg::VecA(T), obs::VecA(T), state_cov::CovM(T), H_obs::Function,
        obs_cov::CovM(T), kwargs::StepKwargs) where T <: Float64

Computes the local minima of the three-dimension variational cost function with a static 
background covariance using a simple Newton optimization method

`x_bkg` is the initial state proposal vector, `obs` is to the observation vector, 
`state_cov` is the background error covariance matrix, `H_obs` is a model mapping operator
for observations, `obs_cov` is the observation error covariance matrix, and `kwargs` refers
to any additional arguments needed for the operation computation.

```
return  x
```
"""
function D3_var_NewtonOp(x_bkg::VecA(T), obs::VecA(T), state_cov::CovM(T), H_obs::Function,
                         obs_cov::CovM(T), kwargs::StepKwargs) where T <: Float64 

    # initializations
    j_max = 40
    tol = 0.001
    j = 1
    sys_dim = length(x_bkg)

    # first guess is copy of the first background
    x = copy(x_bkg)

    # gradient preallocation over-write
    function grad!(g::VecA(T), x::VecA(T)) where T <: Real
        g[:] = D3_var_grad(x, obs, x_bkg, state_cov, H_obs, obs_cov, kwargs)
    end

    # Hessian preallocation over-write
    function hess!(h::ArView(T), x::VecA(T)) where T <: Real
        h .= D3_var_hessian(x, obs, x_bkg, state_cov, H_obs, obs_cov, kwargs)
    end

    # perform the optimization by simple Newton
    grad_x = Array{Float64}(undef, sys_dim)
    hess_x = Array{Float64}(undef, sys_dim, sys_dim)

    while j <= j_max
        # compute the gradient and Hessian
        grad!(grad_x, x)
        hess!(hess_x, x)

        # perform Newton approximation
        Δx = inv(hess_x) * grad_x
        x = x - Δx

        if norm(Δx) < tol
            break
        else
            j+=1
        end
    end
    return x
end


##############################################################################################
"""
    D4_var_cost_lag1(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
        H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real

Computes the cost of the lag-1 four-dimensional variational analysis increment from an initial
state proposal with a static background covariance

`x` is a free argument used to evaluate the cost of the given state proposal versus other 
proposal states, `obs` is to the observation vector, `x_bkg` is the initial state proposal 
vector, `state_cov` is the background error covariance matrix, `H_obs` is a model mapping
operator for observations, and `obs_cov` is the observation error covariance matrix.
`kwargs` refers to any additional arguments needed for the operation computation.

```
return  0.5*back_component + 0.5*obs_component
```
"""
function D4_var_lag1_cost(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
                     H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real

    # initializations
    obs_dim = length(obs)

    # dummy time variable for RK scheme
    t = 0.0001

    f_steps = kwargs["f_steps"]

    # compute increment with background term versus free variable x (weighted with B)
    # background discrepancy
    δ_b = x - x_bkg
    # cost function
    back_component = dot(δ_b, inv(state_cov) * δ_b)

    for i in 1:f_steps
        rk4_step!(x, t, kwargs)
    end

    # compute the increment of the new x (evolved in time) with the observation y (weighted with R)
    # compute the evolution of x under the RK scheme
    H = H_obs(x, obs_dim, kwargs)
    # observation discrepancy
    δ_o = obs - H
    # cost function
    obs_component = dot(δ_o, inv(obs_cov) * δ_o)

    0.5*back_component + 0.5*obs_component
end


##############################################################################################
"""
    D4_var_lag1_grad(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
        H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Float64

Computes the gradient of the lag-1 four-dimensional variational analysis increment from an initial 
state proposal with a static background covariance using a wrapper function for automatic 
differentiation

`x` is a free argument used to evaluate the cost of the given state proposal versus other 
proposal states, `obs` is to the observation vector, `x_bkg` is the initial state proposal 
vector, `state_cov` is the background error covariance matrix, `H_obs` is a model mapping
operator  for observations, and `obs_cov` is the observation error covariance matrix.
`kwargs` refers to any additional arguments needed for the operation computation.

`wrap_cost` is a function that allows differentiation with respect to the free argument `x`
while treating all other hyperparameters of the cost function as constant.

```
return  ForwardDiff.gradient(wrap_cost, x)
```
"""
function D4_var_lag1_grad(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
                     H_obs::Function, obs_cov::CovM(T),
                     kwargs::StepKwargs) where T <: Float64

    function wrap_cost(x::VecA(T)) where T <: Real
        D4_var_lag1_cost(x, obs, x_bkg, state_cov, H_obs, obs_cov, kwargs)
    end

    ForwardDiff.gradient(wrap_cost, x)
end


##############################################################################################
"""
    D4_var_lag1_hessian(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
        H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Float64 

Computes the Hessian of the lag-1 four-dimensional analysis increment from an initial 
state proposal with a static background covariance using a wrapper function for automatic 
differentiation

`x` is a free argument used to evaluate the cost of the given state proposal versus other 
proposal states, `obs` is to the observation vector, `x_bkg` is the initial state proposal 
vector, `state_cov` is the background error covariance matrix, `H_obs` is a model mapping 
operator for observations, and `obs_cov` is the observation error covariance matrix.
`kwargs` refers to any additional arguments needed for the operation computation.

`wrap_cost` is a function that allows differentiation with respect to the free argument `x`
while treating all other hyperparameters of the cost function as constant.

```
return  ForwardDiff.hessian(wrap_cost, x)
```
"""
function D4_var_lag1_hessian(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
                        H_obs::Function, obs_cov::CovM(T),
                        kwargs::StepKwargs) where T <: Float64 

    function wrap_cost(x::VecA(T)) where T <: Real
        D4_var_lag1_cost(x, obs, x_bkg, state_cov, H_obs, obs_cov, kwargs)
    end

    ForwardDiff.hessian(wrap_cost, x)
end


##############################################################################################
"""
    D4_var_lag1_NewtonOp(x_bkg::VecA(T), obs::VecA(T), state_cov::CovM(T), H_obs::Function,
        obs_cov::CovM(T), kwargs::StepKwargs) where T <: Float64

Computes the local minima of the lag-1 four-dimensional cost function with a static 
background covariance using a simple Newton optimization method

`x_bkg` is the initial state proposal vector, `obs` is to the observation vector, 
`state_cov` is the background error covariance matrix, `H_obs` is a model mapping operator
for observations, `obs_cov` is the observation error covariance matrix, and `kwargs` refers
to any additional arguments needed for the operation computation.

```
return  x
```
"""
function D4_var_lag1_NewtonOp(x_bkg::VecA(T), obs::VecA(T), state_cov::CovM(T), H_obs::Function,
                         obs_cov::CovM(T), kwargs::StepKwargs) where T <: Float64 

    # initializations
    j_max = 40
    tol = 0.001
    j = 1
    sys_dim = length(x_bkg)

    # first guess is copy of the first background
    x = copy(x_bkg)

    # gradient preallocation over-write
    function grad!(g::VecA(T), x::VecA(T)) where T <: Real
        g[:] = D4_var_lag1_grad(x, obs, x_bkg, state_cov, H_obs, obs_cov, kwargs)
    end

    # Hessian preallocation over-write
    function hess!(h::ArView(T), x::VecA(T)) where T <: Real
        h .= D4_var_lag1_hessian(x, obs, x_bkg, state_cov, H_obs, obs_cov, kwargs)
    end

    # perform the optimization by simple Newton
    grad_x = Array{Float64}(undef, sys_dim)
    hess_x = Array{Float64}(undef, sys_dim, sys_dim)

    while j <= j_max
        # compute the gradient and Hessian
        grad!(grad_x, x)
        hess!(hess_x, x)

        # perform Newton approximation
        Δx = inv(hess_x) * grad_x
        x = x - Δx

        if norm(Δx) < tol
            break
        else
            j+=1
        end
    end
    return x
end


##############################################################################################
"""
D4_var_lagL_cost(x::AbstractVector{T1}, obs::ArView(T2), x_bkg::VecA(T2), state_cov::CovM(T2),
                H_obs::Function, obs_cov::CovM(T2), L::Int64, S::Int64, 
                kwargs::StepKwargs) where T1 <: Real where T2 <: Float64

Computes the cost of the lag-L four-dimensional variational analysis increment with a data 
assimilation window of shift S from an initial state proposal with a static background 
covariance.

`x` is a free argument used to evaluate the cost of the given state proposal versus other 
proposal states, `obs` is to the observation vector, `x_bkg` is the initial state proposal 
vector, `state_cov` is the background error covariance matrix, `H_obs` is a model mapping
operator for observations, and `obs_cov` is the observation error covariance matrix, `L` 
specifies the lag, and `S` specifies the shift. `kwargs` refers to any additional 
arguments needed for the operation computation.

```
return  0.5*back_component + 0.5*obs_component
```
"""
function D4_var_lagL_cost(x::AbstractVector{T1}, obs::ArView(T2), x_bkg::VecA(T2), state_cov::CovM(T2),
                          H_obs::Function, obs_cov::CovM(T2), L::Int64, S::Int64, kwargs::StepKwargs) where T1 <: Real where T2 <: Float64
    
    # check that shift is valid
    if S < 1
        throw(ArgumentError("Shift must be greater than or equal to 1. Received input shift = $S."))
    end

    # handle covariance matrices and inputs properly for Dual types
    state_cov_inv = T1 <: ForwardDiff.Dual ? inv(convert(Diagonal{T1}, state_cov)) : inv(state_cov)
    obs_cov_inv = T1 <: ForwardDiff.Dual ? inv(convert(Diagonal{T1}, obs_cov)) : inv(obs_cov)

    # initializations
    # dimension of individual observation vectors
    obs_dim = size(obs, 1)
    # dummy time variable for RK4 scheme
    t = 0.0001       
    # check that f_steps is in kwargs
    f_steps = get(kwargs, "f_steps", 1)::Int64

    # check that lag doesn't exceed the number of observations in the obs array
    max_obs_time = size(obs, ndims(obs))  # Get the max number of time steps available
    if L > max_obs_time
        throw(ArgumentError("Lag ($L) cannot exceed the number of available observation time steps ($max_obs_time)."))
    end

    # compute background discrepancy
    δ_b = x - x_bkg
    back_component = dot(δ_b, state_cov_inv * δ_b)

    obs_component = 0
    x_new = copy(x)

    # loop bounds incorporate data assimilation window shift
    for k in 1:S:L
        # break loop if k exceeds obs's available time steps
        if k > max_obs_time
            break
        end

        # Evolve the state over f_steps using RK4
        for i in 1:f_steps
            rk4_step!(x_new, t, kwargs)
        end

        # apply observation operator at each step
        H = H_obs(x_new, obs_dim, kwargs)

        # handle 2D and 3D observation arrays
        if ndims(obs) == 2
            δ_o = obs[:, k] - H
        elseif ndims(obs) == 3
            δ_o = obs[:, :, k] - H
        else
            throw(DimensionMismatch("obs must be a 2D or 3D array."))
        end

        # compute observation component with observation covariance inverse
        obs_component += dot(δ_o, obs_cov_inv * δ_o)
    end

    return 0.5 * back_component + 0.5 * obs_component
end


##############################################################################################
"""
D4_var_lagL_grad(x::AbstractVector{T}, obs::ArView(T), x_bkg::VecA(T), state_cov::CovM(T),
                     H_obs::Function, obs_cov::CovM(T), L::Int64, S::Int64, 
                     kwargs::StepKwargs) where T <: Float64 

Computes the cost of the lag-L four-dimensional variational analysis increment with a data 
assimilation window of shift S from an initial state proposal with a static background 
covariance using a wrapper function for automatic differentiation.

`x` is a free argument used to evaluate the cost of the given state proposal versus other 
proposal states, `obs` is to the observation vector, `x_bkg` is the initial state proposal 
vector, `state_cov` is the background error covariance matrix, `H_obs` is a model mapping
operator  for observations, and `obs_cov` is the observation error covariance matrix, and `L` 
specifies the lag, and `S` specifies the shift. `kwargs` refers to any additional arguments 
needed for the operation computation.

`wrap_cost` is a function that allows differentiation with respect to the free argument `x`
while treating all other hyperparameters of the cost function as constant.

```
return  ForwardDiff.gradient(wrap_cost, x)
```
"""
function D4_var_lagL_grad(x::AbstractVector{T}, obs::ArView(T), x_bkg::VecA(T), state_cov::CovM(T),
                     H_obs::Function, obs_cov::CovM(T), L::Int64, S::Int64, 
                     kwargs::StepKwargs) where T <: Float64 

    # Cast covariance matrices if Dual numbers are detected
    if T <: ForwardDiff.Dual
        state_cov = Diagonal(ForwardDiff.Dual.(vec(diagm(state_cov))))
        obs_cov = Diagonal(ForwardDiff.Dual.(vec(diagm(obs_cov))))
    end

    function wrap_cost(x::VecA(T)) where T <: Real
        D4_var_lagL_cost(x, obs, x_bkg, state_cov, H_obs, obs_cov, L, S, kwargs)
    end

    ForwardDiff.gradient(wrap_cost, x)
end


##############################################################################################
"""
D4_var_lagL_hessian(x::AbstractVector{T}, obs::ArView(T), x_bkg::VecA(T), state_cov::CovM(T),
                        H_obs::Function, obs_cov::CovM(T), L::Int64, S::Int64,
                        kwargs::StepKwargs) where T <: Float64

Computes the Hessian of the lag-L four-dimensional variational analysis increment with a data 
assimilation window of shift S from an initial state proposal with a static background covariance 
using a wrapper function for automatic differentiation.

`x` is a free argument used to evaluate the cost of the given state proposal versus other 
proposal states, `obs` is to the observation vector, `x_bkg` is the initial state proposal 
vector, `state_cov` is the background error covariance matrix, `H_obs` is a model mapping 
operator for observations, `obs_cov` is the observation error covariance matrix, and `L` 
specifies the lag, and `S` specifies the shift. `kwargs` refers to any additional arguments 
needed for the operation computation.

`wrap_cost` is a function that allows differentiation with respect to the free argument `x`
while treating all other hyperparameters of the cost function as constant.

```
return  ForwardDiff.hessian(wrap_cost, x)
```
"""
function D4_var_lagL_hessian(x::AbstractVector{T}, obs::ArView(T), x_bkg::VecA(T), state_cov::CovM(T),
                        H_obs::Function, obs_cov::CovM(T), L::Int64, S::Int64,
                        kwargs::StepKwargs) where T <: Float64

    function wrap_cost(x::VecA(T)) where T <: Real
        D4_var_lagL_cost(x, obs, x_bkg, state_cov, H_obs, obs_cov, L, S, kwargs)
    end

    ForwardDiff.hessian(wrap_cost, x)
end


##############################################################################################
"""
    D4_var_lagL_NewtonOp(x_bkg::AbstractVector{T}, obs::ArView(T), state_cov::CovM(T), H_obs::Function,
                         obs_cov::CovM(T), L::Int64, S::Int64, kwargs::StepKwargs) where T <: Float64

Computes the local minima of the lag-L four-dimensional variational analysis increment with a data 
assimilation window of shift S with a static background covariance using a simple Newton optimization 
method.

`x_bkg` is the initial state proposal vector, `obs` is to the observation vector, 
`state_cov` is the background error covariance matrix, `H_obs` is a model mapping operator
for observations, `obs_cov` is the observation error covariance matrix, `L` 
specifies the lag, and `S` specifies the shift. `kwargs` refers to any additional 
arguments needed for the operation computation.
```
return  x
```
"""
function D4_var_lagL_NewtonOp(x_bkg::AbstractVector{T}, obs::ArView(T), state_cov::CovM(T), H_obs::Function,
                         obs_cov::CovM(T), L::Int64, S::Int64, kwargs::StepKwargs) where T <: Float64

    # initializations
    j_max = 40
    tol = 0.001
    j = 1
    sys_dim = length(x_bkg)

    # first guess is copy of the first background
    x = copy(x_bkg)

    # gradient preallocation over-write
    function grad!(g::VecA(T), x::VecA(T)) where T <: Real
        g[:] = D4_var_lagL_grad(x, obs, x_bkg, state_cov, H_obs, obs_cov, L, S, kwargs)
    end

    # Hessian preallocation over-write
    function hess!(h::ArView(T), x::VecA(T)) where T <: Real
        h .= D4_var_lagL_hessian(x, obs, x_bkg, state_cov, H_obs, obs_cov, L, S, kwargs)
    end

    # perform the optimization by simple Newton
    grad_x = Array{Float64}(undef, sys_dim)
    hess_x = Array{Float64}(undef, sys_dim, sys_dim)

    while j <= j_max
        # compute the gradient and Hessian
        grad!(grad_x, x)
        hess!(hess_x, x)

        # perform Newton approximation
        Δx = inv(hess_x) * grad_x
        x = x - Δx

        if norm(Δx) < tol
            break
        else
            j+=1
        end
    end
    return x
end

##############################################################################################
# end module

end
