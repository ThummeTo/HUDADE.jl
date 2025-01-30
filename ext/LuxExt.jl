#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module LuxExt

using HUDADE, Lux

import HUDADE.Random
using HUDADE.ComponentArrays
import HUDADE: NoVector, undual

# evaluates the Lux model (oop)
function evalLux(model, u, p, st)
    #@info "p: $(p)"  
    #@info "model(u, p, st): $(model(u, p, st))"
    #@info "evalLux(u, p, st): $(evalLux(u, p, st))"

    #@info "$(typeof(u)), $(typeof(p)), $(typeof(st))"

    return first(model(u, p, st))
end

# evaluates the Lux model (ip)
# ToDo: is there an inplace option?
function evalLux!(y, model, u, p, st)
    y[:] = evalLux(u, p, st)
    return nothing
end

"""
[ToDo]
"""
function HUDADE.LuxModel(model; tspan=(0.0, 1.0))

    rng = Random.default_rng()
    _p, st = Lux.setup(rng, model)

    p = Float64.(ComponentArray(_p))

    # functions 
    g = function(y, x_c, x_d, u, p, t)
        evalLux!(y, model, u, p, st)
        nothing
    end
    
    fct = HUDAODEFunction(; g=g)
    return HUDAODEProblem(fct, NoVector, NoVector, tspan; p=p)
end

"""
[ToDo]
"""
function HUDADE.LuxSecondOrderModel(model; tspan=(0.0, 1.0))

    rng = Random.default_rng()
    _p, st = Lux.setup(rng, model)

    p = Float64.(ComponentArray(_p))

    # functions 
    g = function(y, x_c, x_d, u, p, t)
        
        len = round(Int, length(u)/2)

        y_so = evalLux(model, u, p, st)

        for i in 1:len 
            y[1+(i-1)*2] = u[1+(i-1)*2]
            y[i*2] = y_so[i]
        end

        nothing
    end
    
    fct = HUDAODEFunction(; g=g)
    return HUDAODEProblem(fct, NoVector, NoVector, tspan; p=p)
end

"""
   LuxDiscreteModel(...)
   
eventType == :state (augment continuous state for time sampling)
eventType == :time (augment discrete state for sampling)

"""
function HUDADE.LuxDiscreteModel(model, _x_d0::AbstractVector{T}, sampling_freq::T; 
    tspan=(T(0.0), T(1.0)),
    eventType::Symbol = :time,
    problemkwargs...) where {T}

    @assert eventType ∈ (:state, :time) "eventType must be one of (:state, :time)"
    @assert typeof(tspan[1]) == T "tspan[1] is not of type $(T)"
    @assert typeof(tspan[end]) == T "tspan[end] is not of type $(T)"
    @assert typeof(_x_d0[1]) == T "x_d0 is not of type $(T)"

    rng = Random.default_rng()
    _p, st = Lux.setup(rng, model)

    p = ComponentArray(_p)
    if typeof(p[:layer_1][:weight][1]) != T
        p = T.(p)
    end

    dt = T(1.0/sampling_freq)

    α = HUDADE.α_NO_FUNCTION
    c_t = HUDADE.c_t_NO_FUNCTION
    a_t = HUDADE.a_t_NO_FUNCTION
    f = HUDADE.f_NO_FUNCTION
    c_x = HUDADE.c_x_NO_FUNCTION
    a_x = HUDADE.a_x_NO_FUNCTION

    x_c0 = [] 
    x_d0 = [_x_d0...]
    z_len = 0

    if eventType == :time
        x_d0 = [_x_d0..., T(0.0)]

        # functions 
        α = function(x_c, x_d, u, p, t)
            @assert length(x_c) == 0 "LuxDiscreteModel initializes with continuous state $(length(x_c)) != 0."
            @assert length(x_d) == length(x_d0) "LuxDiscreteModel initializes with discrete state $(length(x_d)) != $(length(x_d0))."
            x_d[1:end-1] = _x_d0 
            x_d[end] = dt
            nothing
        end

        c_t = function(x_d, u, p, t) # t_next, 
            #t_next[1] = x_d[end] # undual(x_d[end])
            return undual(x_d[end])
        end

        a_t = function(x_c_right, x_d_right, x_c_left, x_d_left, u, p, t, handle)
            if handle
                x_d_right[1:end-1] = evalLux(model, x_d_left[1:end-1], p, st)
                x_d_right[end] = x_d_left[end] + dt
            end
            nothing
        end
    else # if eventType == :state
        x_c0 = [0.0]
        z_len = 1

        # functions 
        α = function(x_c, x_d, u, p, t)
            @assert length(x_c) == 1 "LuxDiscreteModel initializes with continuous state $(length(x_c)) != 1."
            @assert length(x_d) == length(x_d0) "LuxDiscreteModel initializes with discrete state $(length(x_d)) != $(length(x_d0))."
            x_d[:] = x_d0 
            nothing
        end

        f = function(ẋ_c, x_c, x_d, u, p, t)
            ẋ_c[1] = 1.0 # approximates time as state :-)
            nothing
        end

        c_x = function(z, x_c, x_d, u, p)
            z[1] = dt - x_c[1]
            nothing
        end

        a_x = function(x_c_right, x_d_right, x_c_left, x_d_left, u, p, t, idx)
            if idx > 0
                x_d_right[:] = evalLux(model, x_d_left, p, st)
                x_c_right[1] = 0.0
            end
            nothing
        end
    end

    fct = HUDAODEFunction(; 
        x_c_len=length(x_c0),
        x_d_len=length(x_d0), α=α, c_t=c_t, a_t=a_t, f=f, c_x=c_x, a_x=a_x, z_len=z_len)
    return HUDAODEProblem(fct, T[], [x_c0..., x_d0...], tspan; 
        p=p, problemkwargs...)
end

function HUDADE.LuxNeuralODE(model, x_c0; tspan=(0.0, 1.0))

    rng = Random.default_rng()
    _p, st = Lux.setup(rng, model)

    p = Float64.(ComponentArray(_p))

    # functions 
    f = function(ẋ_c, x_c, x_d, u, p, t)
        evalLux!(ẋ_c, model, x_c, p, st)
        nothing
    end
    
    fct = HUDAODEFunction(; x_c_len=length(x_c0), f=f)
    return HUDAODEProblem(fct, x_c0, NoVector, tspan; p=p)
end

function HUDADE.LuxSecondOrderNeuralODE(model, x_c0; tspan=(0.0, 1.0))

    rng = Random.default_rng()
    _p, st = Lux.setup(rng, model)

    p = Float64.(ComponentArray(_p))

    # functions 
    f = function(ẋ_c, x_c, x_d, u, p, t)
        len = round(Int, length(x_c)/2)

        ẍ_c = evalLux(model, x_c, p, st)

        for i in 1:len 
            ẋ_c[1+(i-1)*2] = x_c[i*2]
            ẋ_c[i*2] = ẍ_c[i]
        end

        nothing
    end
    
    fct = HUDAODEFunction(; x_c_len=length(x_c0), f=f)
    return HUDAODEProblem(fct, x_c0, NoVector, tspan; p=p)
end

end # LuxExt