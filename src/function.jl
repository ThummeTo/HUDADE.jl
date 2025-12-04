#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import SciMLSensitivity.SciMLBase
import SciMLSensitivity.SciMLBase: ODEFunction
using  SciMLSensitivity.SciMLBase.RecursiveArrayTools

abstract type AbstractHUDAODEFunction end

struct HUDAODEFunction{F, G, CT, CX, AT, AX, Α, Ω, Ε} <: AbstractHUDAODEFunction
 
    # functions
    f::F
    g::G
    c_t ::CT
    c_x::CX
    a_t::AT
    a_x::AX
    α::Α
    ω::Ω
    ϵ::Ε

    x_c_len::Int64
    x_d_len::Int64
    z_len::Int64
    #p::AbstractVector{Float64}

    # pack, unpack  [ToDo: types!]
    p_pack 
    p_unpack
    x_pack 
    x_unpack

    function HUDAODEFunction{F, G, CT, CX, AT, AX, Α, Ω, Ε}(;
        f::F=f_NO_FUNCTION,
        g::G=g_NO_FUNCTION,
        c_t::CT=c_t_NO_FUNCTION,
        c_x::CX=c_x_NO_FUNCTION,
        a_t::AT=a_t_NO_FUNCTION,
        a_x::AX=a_x_NO_FUNCTION,
        α::Α=α_NO_FUNCTION,
        ω::Ω=ω_NO_FUNCTION,
        ϵ::Ε=ϵ_NO_FUNCTION,
        x_c_len::Int64=0,
        x_d_len::Int64=0,
        z_len::Int64=0,
        #p::AbstractVector{Float64}=NoVector,
        p_pack=nothing, # TODO TYPE!
        p_unpack=nothing, # TODO TYPE!
        x_pack=nothing, # TODO TYPE!
        x_unpack=nothing) where {F, G, CT, CX, AT, AX, Α, Ω, Ε} 

        # size / cache
        #p = copy(p)

        p_pack = nothing 
        p_unpack = nothing

        u_c = Float64[]
        u_d = Float64[]

        x_pack = function(_x_c, _x_d, _u_c, _u_d)
            return vcat(_x_c, _x_d, _u_c, _u_d) #ArrayPartition
        end
        x_unpack = function(container)
            u_c_len = length(u_c)  # ToDo: views!
            return (container[1:x_c_len],
                container[x_c_len+1:x_c_len+x_d_len],
                container[x_c_len+x_d_len+1:x_c_len+x_d_len+u_c_len],
                container[x_c_len+x_d_len+u_c_len+1:end])
        end
        
        return new{F, G, CT, CX, AT, AX, Α, Ω, Ε}(f, g, c_t, c_x, a_t, a_x, α, ω, ϵ, x_c_len, x_d_len, z_len, p_pack, p_unpack, x_pack, x_unpack)
    end

    function HUDAODEFunction(;f::F=f_NO_FUNCTION,
        g::G=g_NO_FUNCTION,
        c_t::CT=c_t_NO_FUNCTION,
        c_x::CX=c_x_NO_FUNCTION,
        a_t::AT=a_t_NO_FUNCTION,
        a_x::AX=a_x_NO_FUNCTION,
        α::Α=α_NO_FUNCTION,
        ω::Ω=ω_NO_FUNCTION,
        ϵ::Ε=ϵ_NO_FUNCTION,
        kwargs...) where {F, G, CT, CX, AT, AX, Α, Ω, Ε} 

        return HUDAODEFunction{F, G, CT, CX, AT, AX, Α, Ω, Ε}(;f=f, g=g, c_t=c_t, c_x=c_x, a_t=a_t, a_x=a_x, α=α, ω=ω, ϵ=ϵ, kwargs...)
    end
end
export HUDAODEFunction

function rebuild(fct::HUDAODEFunction; 
    f::F=fct.f,
    g::G=fct.g,
    c_t::CT=fct.c_t,
    c_x::CX=fct.c_x,
    a_t::AT=fct.a_t,
    a_x::AX=fct.a_x,
    α::Α=fct.α,
    ω::Ω=fct.ω,
    ϵ::Ε=fct.ϵ,
    x_c_len::Int64=fct.x_c_len,
    x_d_len::Int64=fct.x_d_len,
    z_len::Int64=fct.z_len,
    #p::AbstractVector{Float64}=fct.p,
    p_pack=fct.p_pack, # TODO TYPE!
    p_unpack=fct.p_unpack, # TODO TYPE!
    x_pack=fct.x_pack, # TODO TYPE!
    x_unpack=fct.x_unpack) where {F, G, CT, CX, AT, AX, Α, Ω, Ε} # TODO TYPE!

    return HUDAODEFunction{F, G, CT, CX, AT, AX, Α, Ω, Ε}(;f=f, g=g, c_t=c_t, c_x=c_x, a_t=a_t, a_x=a_x, α=α, ω=ω, ϵ=ϵ, 
        x_c_len=x_c_len, x_d_len=x_d_len, z_len=z_len, p_pack=p_pack, p_unpack=p_unpack, x_pack=x_pack, x_unpack=x_unpack)
end

function augment_discrete_state(fct::HUDAODEFunction, aug_x_d)
    x_d_len = fct.x_d_len + length(aug_x_d)
    return rebuild(fct, x_d_len=x_d_len)
end

function SciMLBase.solve(fct::HUDAODEFunction, x0_c, x0_d, tspan, args...; p=HUDADE.NoVector, kwargs...)

    # [ToDo] this needs improvement
    tType = Float64 
    @warn "Please create a HUDADEProblem for solving ..."
    # if length(p) > 0
    #     tType = typeof(p[1])
    # elseif length(x0_c) > 0
    #     tType = typeof(x0_c[1])
    # elseif length(x0_d) > 0
    #     tType = typeof(x0_d[1])
    # end

    #@info "tType=$(tType)"
    problem = HUDAODEProblem(fct, x0_c, x0_d, tspan; p=p, tType=tType)
    return SciMLBase.solve(problem, args...; p=p, kwargs...)
end