#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Revise
using HUDADE, Lux

using Test
using DifferentialEquations
using Plots, Plots.Measures, Colors, LaTeXStrings

using LinearAlgebra
using ComponentArrays, Random
using ProgressMeter
using Optimization, OptimizationOptimisers

using ReverseDiff, ForwardDiff, FiniteDiff

import Random 
Random.seed!(1234)

function plotBB(args...; textsize=12, kwargs...)
    
    return plot(args...; size=(600,600), dpi=300, 
        xtickfontsize=textsize, 
        ytickfontsize=textsize, 
        titlesize=textsize, 
        legendfontsize=textsize, 
        xguidefontsize=textsize, 
        yguidefontsize=textsize, 
        xlabel=L"s_x [m]", 
        ylabel=L"s_y [m]", 
        xlims=(-1.15, 1.15),
        ylims=(-1.15, 1.15),
        kwargs...)
end

function plotCC(args...; textsize=12, kwargs...)
    
    return plot(args...; size=(600,600), dpi=300, 
        xtickfontsize=textsize, 
        ytickfontsize=textsize, 
        titlesize=textsize, 
        legendfontsize=textsize, 
        xguidefontsize=textsize, 
        yguidefontsize=textsize, 
        xlabel="Training horizon [%]", 
        ylabel="Correlation Coefficient [/]", 
        left_margin=5Plots.mm,
        kwargs...)
end

function scatterBB!(fig, sol=solution_gt[1]; scatterlabels = [0.0, 0.8, 1.4, 1.9, 2.3], textsize=12)
    for lab in scatterlabels
        x, _, y, _, _ = sol(lab)
        scatter!(fig, [x], [y], label=:none, color=:black)

        y -= 0.06 # add a little offset

        # from: https://discourse.julialang.org/t/how-to-change-annotate-background-color/40606/2
        Δx = 0.18
        Δy = 0.08
        box = Plots.Shape(:rect)
	    Plots.scale!(box, Δx, Δy)
	    Plots.translate!(box, x, y-Δy/2)
	    Plots.plot!(fig, box, c=:white, linestroke=:black, label=false)

        annotate!(fig, x, y, Plots.text("$(round(lab; digits=2))s", :black, :top, textsize))
    end
    return fig
end

function plotMatrix!(fig, A; prefix="", kwargs...)

    A = abs.(A)
    n = max(A...)

    if n == 0.0
        n = 1.0
    end

    if n < 1.0
        n = 1.0
    end

    scaleStr = "$(round(1.0/n; digits=3))"

    plot!(fig, Gray.(A ./ n); kwargs..., title=L"%$(prefix)\times %$(scaleStr)", xrotation=90) # xmirror=true, 
end

function shortString(str)
    out = ""
    if parse(Int, str[2]) == 1
        out *= "P"
    end
    if parse(Int, str[4]) == 1
        out *= "S"
    end
    if parse(Int, str[6]) == 1
        out *= "F"
    end
    if parse(Int, str[8]) == 1
        out *= "O"
    end
    return out
end

"""

    unsense(a)
    
Converts AD-primitives `a` to Float - if necessary.
Works for vectors, too.
"""
function unsense(a)
    return a 
end
function unsense(a::ForwardDiff.Dual)
    return ForwardDiff.value(a)
end
function unsense(a::AbstractArray)
    return collect(unsense(e) for e in a)
end

"""
The combined system of equations w.o. algebraic loops by design.
"""
mutable struct CombinedEquationSystem
    f_a 
    f_b 

    W_az::Matrix
    W_ba::Matrix
    W_bz::Matrix
    W_za::Matrix
    W_zb::Matrix
    W_zz::Matrix

    b_a::Vector
    b_b::Vector
    b_z::Vector

    γ_a::Vector
    γ_b::Vector

    last_t 

    parallel::Bool
    sequential::Bool
    feedthrough::Bool

    len_u
    len_u_a
    len_u_b
    len_y
    len_y_a 
    len_y_b

    function CombinedEquationSystem(f_a, f_b; 
        len_u, 
        len_u_a, 
        len_u_b, 
        len_y, 
        len_y_a=-1, 
        len_y_b=-1)

        inst = new()

        inst.parallel = false
        inst.sequential = false
        inst.feedthrough = false

        inst.f_a = f_a 
        inst.f_b = f_b

        inst.len_u = len_u
        inst.len_u_a = len_u_a
        inst.len_u_b = len_u_b
        inst.len_y = len_y

        if len_y_a == -1
            inst.len_y_a = length(f_a(zeros(len_u_a)))
        else
            inst.len_y_a = len_y_a
        end

        if len_y_b == -1
            inst.len_y_b = length(f_b(zeros(len_u_b)))
        else
            inst.len_y_b = len_y_b
        end

        # inst.W_aa = zeros(len_u_a, len_y_a)
        # inst.W_ab = zeros(len_u_a, len_y_b)
        inst.W_az = zeros(len_u_a, len_u)

        inst.W_ba = zeros(len_u_b, len_y_a)
        # inst.W_bb = zeros(len_u_b, len_y_b)
        inst.W_bz = zeros(len_u_b, len_u)

        inst.W_za = zeros(len_y, len_y_a)
        inst.W_zb = zeros(len_y, len_y_b)
        inst.W_zz = zeros(len_y, len_u)

        inst.γ_a = zeros(Real, len_y_a)
        inst.γ_b = zeros(Real, len_y_b)

        inst.b_a = zeros(Real, len_u_a)
        inst.b_b = zeros(Real, len_u_b)
        inst.b_z = zeros(Real, len_y)

        return inst
    end
end

function getParameters(sys::CombinedEquationSystem; train::Bool=true, bias::Bool=false)

    names = []
    values = []

    if train
        initialize!(sys)
    else
        connect!(sys)
    end

    if sys.parallel 
        if :W_az ∉ names
            push!(names, :W_az)
            push!(values, sys.W_az)
        end
        if :W_bz ∉ names
            push!(names, :W_bz)
            push!(values, sys.W_bz)
        end
        if :W_za ∉ names
            push!(names, :W_za)
            push!(values, sys.W_za)
        end
        if :W_zb ∉ names
            push!(names, :W_zb)
            push!(values, sys.W_zb)
        end
    end

    if sys.sequential
        if :W_az ∉ names
            push!(names, :W_az)
            push!(values, sys.W_az)
        end
        if :W_ba ∉ names
            push!(names, :W_ba)
            push!(values, sys.W_ba)
        end
        if :W_zb ∉ names
            push!(names, :W_zb)
            push!(values, sys.W_zb)
        end
    end

    if sys.feedthrough
        if :W_zz ∉ names
            push!(names, :W_zz)
            push!(values, sys.W_zz)
        end
    end

    if bias
        if :b_a ∉ names
            push!(names, :b_a)
            push!(values, sys.b_a)
        end
        if :b_b ∉ names
            push!(names, :b_b)
            push!(values, sys.b_b)
        end
        if :b_z ∉ names
            push!(names, :b_z)
            push!(values, sys.b_z)
        end
    end

    nt = NamedTuple()
    if train
        names = (names...,)
        values = collect((values=Float64[val...],) for val in values)
        nt = NamedTuple{names}(values)
    end

    return nt
end

function identifiy!(A)
    r, c = size(A)
    for i in 1:r
        A[i,i] = 1.0
    end
    nothing
end

function zerofiy!(A)
    A .= 0.0
    nothing
end

function randomfiy!(A, noise)
    r, c = size(A)
    for i in 1:r
        for j in 1:c
            A[i, j] = (-1.0 + rand()*2.0) * noise
        end
    end
    nothing
end

function connect!(sys::CombinedEquationSystem, parallel::Bool=sys.parallel, sequential::Bool=sys.sequential, feedthrough::Bool=sys.feedthrough)

    sys.parallel = parallel
    sys.sequential = sequential
    sys.feedthrough = feedthrough

    # resolve all connections
    zerofiy!(sys.W_az)
    zerofiy!(sys.W_ba)
    zerofiy!(sys.W_bz)
    zerofiy!(sys.W_za)
    zerofiy!(sys.W_zb)
    zerofiy!(sys.W_zz)

    # resert bias
    zerofiy!(sys.b_a)
    zerofiy!(sys.b_b)
    zerofiy!(sys.b_z)

    if sys.parallel 
        identifiy!(sys.W_az)
        identifiy!(sys.W_bz)
        identifiy!(sys.W_za)
        identifiy!(sys.W_zb)
    end

    if sys.sequential
        identifiy!(sys.W_az)
        identifiy!(sys.W_ba)
        identifiy!(sys.W_zb)
    end

    if sys.feedthrough
        identifiy!(sys.W_zz)
    end

    nothing
end

function initialize!(sys::CombinedEquationSystem;
    noise::Real=1e-4)

    # resolve all connections
    zerofiy!(sys.W_az)
    zerofiy!(sys.W_ba)
    zerofiy!(sys.W_bz)
    zerofiy!(sys.W_za)
    zerofiy!(sys.W_zb)
    zerofiy!(sys.W_zz)

    # resert bias
    zerofiy!(sys.b_a)
    zerofiy!(sys.b_b)
    zerofiy!(sys.b_z)

    if sys.parallel 
        randomfiy!(sys.W_az, noise)
        randomfiy!(sys.W_bz, noise)
        randomfiy!(sys.W_za, noise)
        randomfiy!(sys.W_zb, noise)
        
        identifiy!(sys.W_az)
        identifiy!(sys.W_za)
    end

    if sys.sequential
        randomfiy!(sys.W_az, noise)
        randomfiy!(sys.W_ba, noise)
        randomfiy!(sys.W_zb, noise)

        identifiy!(sys.W_az)
        identifiy!(sys.W_ba)

        if !sys.parallel
            identifiy!(sys.W_zb)
        end
    end

    if sys.feedthrough
        randomfiy!(sys.W_zz, noise)
    end

    nothing
end

# the general case would be `eval!(sys, γ, υ)` but we do some simplifications here
function eval!(sys::CombinedEquationSystem, γ::AbstractArray{<:Real}, υ::AbstractArray{<:Real}, x_d, u, p, t)

    if hasproperty(p, :W_az) 
        sys.W_az = reshape(p[:W_az][:values], sys.len_u_a, sys.len_u)
    end
    if hasproperty(p, :W_ba)
        sys.W_ba = reshape(p[:W_ba][:values], sys.len_u_b, sys.len_y_a)
    end
    if hasproperty(p, :W_bz)
        sys.W_bz = reshape(p[:W_bz][:values], sys.len_u_b, sys.len_u)
    end
    if hasproperty(p, :W_za) 
        sys.W_za = reshape(p[:W_za][:values], sys.len_y, sys.len_y_a) 
    end
    if hasproperty(p, :W_zb)
         sys.W_zb = reshape(p[:W_zb][:values], sys.len_y, sys.len_y_b) 
    end
    if hasproperty(p, :W_zz) 
        sys.W_zz = reshape(p[:W_zz][:values], sys.len_y, sys.len_u) 
    end

    if hasproperty(p, :b_a) 
        sys.b_a = p[:b_a][:values] 
    end
    if hasproperty(p, :b_b) 
        sys.b_b = p[:b_b][:values] 
    end
    if hasproperty(p, :b_z) 
        sys.b_z = p[:b_z][:values] 
    end

    υ_a = sys.W_az*υ + sys.b_a
    # the general case would be `f_a(sys.γ_a, υ_a)` but we do some simplifications here
    f_a(sys.γ_a, υ_a, x_d, u, p, t)

    υ_b = sys.W_ba*sys.γ_a + sys.W_bz*υ + sys.b_b
    # the general case would be `f_b(sys.γ_b, υ_b)` but we do some simplifications here
    f_b(sys.γ_b, υ_b, x_d, u, p, t)

    γ[:] = sys.W_za*sys.γ_a + sys.W_zb*sys.γ_b + sys.W_zz*υ + sys.b_z
    nothing
end

function BouncingBall2D_AirFriction(; kwargs...)
    
    gt = HUDADE.BouncingBall2D(; kwargs...)

    gt_f = function(ẋ_c, x_c, x_d, u, p, t)
        sx, vx, sy, vy = x_c 
        m, g, r, d = p 
        n = x_d[1]
    
        air_res = 0.2
    
        ax = 0.0 - sign(vx) * (vx*vx*air_res)
        ay =  -g - sign(vy) * (vy*vy*air_res)
    
        ẋ_c[1] = vx 
        ẋ_c[2] = ax
        ẋ_c[3] = vy
        ẋ_c[4] = ay
        nothing
    end
    
    gt = HUDADE.rebuild(gt; f=gt_f)

    return gt 
end