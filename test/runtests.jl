#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using HUDADE

using Test
using DifferentialEquations
using Plots

using HUDADE.SciMLSensitivity: ForwardDiff, FiniteDiff, ReverseDiff, Zygote

t0 = 0.0
z0 = zeros(2)

x_c0 = [1.0, 0.0]
x_d0 = [0.0, -1.0, 1.0]
tspan = (t0, 5.0)

problem = HUDADE.BouncingBall2D()
p0 = fct.p
#problem = HUDAODEProblem(fct, x_c0, x_d0, tspan; p=p0)
solution = solve(problem)
ts = tspan[1]:0.01:tspan[2]
plot(ts, collect(solution(t)[1] for t in ts))
plot!(ts, collect(solution(t)[2] for t in ts))
plot(collect(solution(t)[1] for t in ts), collect(solution(t)[3] for t in ts))

for i in 1:length(solution.u)
    @test solution.u[i][1] > -1.0
    @test solution.u[i][1] < 1.0
end

function loss(p)

    solution = solve(problem, Tsit5(); tspan=tspan, p=p, saveat=tspan[1]:0.1:tspan[2])

    sum = 0.0
    if isa(solution, ReverseDiff.TrackedArray)
        for i in 1:size(solution)[2]
            sum += solution[1,i]
        end
    else
        for i in 1:length(solution.u)
            sum += solution.u[i][1]
        end
    end
    return sum 
end

params = p0

loss(params)
grad_fwd = ForwardDiff.gradient(loss, params)
grad_fid = FiniteDiff.finite_difference_gradient(loss, params)
grad_rwd = ReverseDiff.gradient(loss, params)
grad_zyg = Zygote.gradient(loss, params)[1]

####

d = HUDAODE.Dense(rand(2,2), rand(2), tanh)