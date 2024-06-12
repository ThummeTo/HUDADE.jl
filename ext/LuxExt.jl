#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module LuxExt

using HUDADE, Lux

import HUDADE.Random
using HUDADE.ComponentArrays
import SciMLSensitivity.ForwardDiff

"""
[ToDo]
"""
function HUDADE.LuxModel(model)

    rng = Random.default_rng()
    _p, st = Lux.setup(rng, model)

    p = Float64.(ComponentArray(_p))

    # functions 
    g = function(y, x_c, x_d, u, p, t)
        y[:] = first(model(u, p, st))
        nothing
    end
    
    return HUDAODEFunction(; p=p, g=g)
end

"""
[ToDo]
"""
function HUDADE.LuxSecondOrderModel(model)

    rng = Random.default_rng()
    _p, st = Lux.setup(rng, model)

    p = Float64.(ComponentArray(_p))

    # functions 
    g = function(y, x_c, x_d, u, p, t)
        
        len = round(Int, length(u)/2)

        y_so = first(model(u, p, st))

        for i in 1:len 
            y[1+(i-1)*2] = u[1+(i-1)*2]
            y[i*2] = y_so[i]
        end

        nothing
    end
    
    return HUDAODEFunction(; p=p, g=g)
end

function HUDADE.LuxDiscreteModel(model, x_d0, sampling_freq::Real=1.0)

    rng = Random.default_rng()
    _p, st = Lux.setup(rng, model)

    p = Float64.(ComponentArray(_p))

    # functions 
    α = function(x_c, x_d, u, p, t)
        x_d[1:end-1] = x_d0 
        x_d[end] = 1.0/sampling_freq
        nothing
    end

    c_t = function(t_next, x_d, u, p, t)
        # [Todo] check if type inconsistency and only FD.value if necessary!
        t_next[1] = ForwardDiff.value(x_d[end])
        nothing
    end

    a_t = function(x_c_right, x_d_right, x_c_left, x_d_left, u, p, t)
        x_d_right[1:end-1] = first(model(x_d_left[1:end-1], p, st))
        x_d_right[end] = x_d_left[end] + 1.0/sampling_freq 
        nothing
    end
    
    return HUDAODEFunction(; x_d_len=length(x_d0)+1, p=p, α=α, c_t=c_t, a_t=a_t)
end

function HUDADE.LuxNeuralODE(model, x_c0)

    rng = Random.default_rng()
    _p, st = Lux.setup(rng, model)

    p = Float64.(ComponentArray(_p))

    # functions 
    f = function(ẋ_c, x_c, x_d, u, p, t)
        ẋ_c[:] = first(model(x_c, p, st))
        nothing
    end
    
    return HUDAODEFunction(; x_c_len=length(x_c0), p=p, f=f)
end

function HUDADE.LuxSecondOrderNeuralODE(model, x_c0)

    rng = Random.default_rng()
    _p, st = Lux.setup(rng, model)

    p = Float64.(ComponentArray(_p))

    # functions 
    f = function(ẋ_c, x_c, x_d, u, p, t)
        len = round(Int, length(x_c)/2)

        ẍ_c = first(model(x_c, p, st))

        for i in 1:len 
            ẋ_c[1+(i-1)*2] = x_c[i*2]
            ẋ_c[i*2] = ẍ_c[i]
        end

        nothing
    end
    
    return HUDAODEFunction(; x_c_len=length(x_c0), p=p, f=f)
end

end # LuxExt