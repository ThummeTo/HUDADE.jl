#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import SciMLSensitivity.SciMLBase: ODESolution

mutable struct HUDADESolution
    solution::ODESolution
    events::Vector{HUDADEEvent}
    
    function HUDADESolution()
        inst =  new()
        inst.events = Vector{HUDADEEvent}()
        return inst
    end
end
export HUDADESolution

function Base.hasproperty(solution::HUDADESolution, var::Symbol)
    return Base.hasfield(HUDADESolution, var) || Base.hasfield(ODESolution, var)
end

function Base.getproperty(solution::HUDADESolution, var::Symbol)
    if Base.hasfield(HUDADESolution, var)
        return Base.getfield(solution, var)
    elseif Base.hasfield(ODESolution, var)
        return Base.getfield(solution.solution, var)
    else
        @assert false "Unknwon field `$(var)`"
    end
end

function Base.setproperty!(solution::HUDADESolution, var::Symbol, value)
    if Base.hasfield(HUDADESolution, var)
        return Base.setfield!(solution, var, value)
    elseif Base.hasfield(ODESolution, var)
        return Base.setfield!(solution.solution, var, value)
    else
        @assert false "Unknwon field `$(var)`"
    end
end

function (solution::HUDADESolution)(args...)
    return solution.solution(args...)
end

function add_event!(solution::HUDADESolution, t::Float64, idx::UInt32)
    event = HUDADEEvent(t, idx)
    push!(solution.events, event)
    return nothing
end
add_event!(solution::HUDADESolution, t::Float64, idx::Integer) = add_event!(solution, t, UInt32(idx))