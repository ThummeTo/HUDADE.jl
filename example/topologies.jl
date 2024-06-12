#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Revise # [ToDo remove]

include(joinpath(@__DIR__, "helpers.jl"))

#    [  m,    g,   r,   d]
p0 = [1.0, 9.81, 0.1, 0.9]
u0 = zeros(Float64, 0)
t0 = 0.0
z0 = zeros(4)
x_c0 = [-0.25, -6.0, 0.5, 8.0]
x_d0 = [0.0] # , 2.0]
tspan = (t0, 2.1)
dt = 0.01
ts = tspan[1]:dt:tspan[2]
x0_test = [0.5, 8.0, -0.25, -6.0, 0.0]

x0 = [[-0.25, -6.0, 0.5, 8.0, 0.0], [0.5, 8.0, -0.25, -6.0, 0.0], [-0.75, 4.0, -0.5, 6.0, 0.0]]

# c_t = function(t_next, x_d, u, p, t)
#     t_next[1] = x_d[2]
#     #@info "tnext: $(t_next[1])"
#     nothing
# end

# a_t = function(x_c_right_local, x_d_right_local, x_c_left, x_d_left, u, p, t)
#     x_c_right_local[:] = x0_test[1:4]
#     x_d_right_local[2] = 1e6
#     #@info "Jump from $(x_c_left) to $(x_c_right_local)"
#     #@info "Jump from $(x_d_left) to $(x_d_right_local)"
#     nothing
# end

# α = function(x_c, x_d, u, p, t)
#     gt_α(x_c, x_d, u, p, t)
#     x_d[2] = 2.0
# end

numSims = 3
solution_gt = Vector{HUDADESolution}(undef, numSims)
solution_fpm = Vector{HUDADESolution}(undef, numSims)

# FPM Test
gt = BouncingBall2D_AirFriction(; x_c0=x_c0)
#gt = HUDADE.augment_discrete_state(gt, [2.0])
#gt_α = gt.α
#gt = HUDADE.rebuild(gt; c_t=c_t, a_t=a_t, α=α)
problem_gt = HUDAODEProblem(gt, x_c0, x_d0, tspan; p=p0)
for i in 1:numSims
    solution_gt[i] = solve(problem_gt; u0=x0[i], saveat=ts)

    fig = plot(collect(u[1] for u in solution_gt[i].u), collect(u[3] for u in solution_gt[i].u))
    display(fig)
end

# splits = collect(e.t for e in solution_gt.events)
# simulations = []
# splitIndex = 1
# _tstart = ts[1]
# _tstop = 0.0
# for i in 1:length(ts)
#     if ts[i] > splits[splitIndex]
#         _tstop = ts[i-1]
#         push!(simulations, (_tstart, _tstop))
#         _tstart = ts[i]
#         splitIndex += 1
#     end

#     if splitIndex > length(splits)
#         _tstop = ts[end]
#         push!(simulations, (_tstart, _tstop))
#         break 
#     end
# end

# start with a fresh one!
fpm = HUDADE.BouncingBall2D(; x_c0=x_c0)
problem_fpm = HUDAODEProblem(fpm, x_c0, x_d0, tspan; p=p0)
for i in 1:numSims
    solution_fpm[i] = solve(problem_fpm; u0=x0[i], saveat=ts)

    fig = plot(collect(u[1] for u in solution_fpm[i].u), collect(u[3] for u in solution_fpm[i].u))
    display(fig)
end

# noisy identity matrix as init for `W`
function init_weight(rng, dims...)
    W = rand(rng, dims...) * 1e-4
    min_dim = min(dims...)
    for i in 1:min_dim 
        W[i, i] += 1.0
    end
    return W 
end

# noise as init for `b`
function init_bias(rng, dims...)
    b = rand(dims...) * 1e-4
    return b 
end

# MLM Test
ann = Chain(Dense(4, 8, tanh), #; init_weight=init_weight, init_bias=init_bias),
            Dense(8, 2, tanh))#, #; init_weight=init_weight, init_bias=init_bias),
            #Dense(8, 2, identity)) #; init_weight=init_weight, init_bias=init_bias)) 
mlm = HUDADE.LuxSecondOrderModel(ann)
mlm_p0 = mlm.p
problem_mlm = HUDAODEProblem(mlm, Float64[], Float64[], tspan; p=mlm_p0)
solution_mlm = solve(problem_mlm; saveat=ts)

# Hybrid model 
y0 = zeros(0)
e0 = zeros(0)

rng = Random.default_rng()
_p, st = Lux.setup(rng, ann)

function f_a(γ_a::AbstractArray{<:Real}, υ_a::AbstractArray{<:Real}, x_d, u, p, t)
    # basically all of {x_c, x_d, u, p, t} should be pulled from υ
    # here, we define all connections identity for {x_d, p, u, t} and pull only {x_c}
    # u=[] in this example
    x_c = υ_a
    fpm_p = p0
    fpm.f(γ_a, x_c, x_d, u, fpm_p, t)
    nothing
end

function f_b(γ_b::AbstractArray{<:Real}, υ_b::AbstractArray{<:Real}, x_d, u, p, t)
    # basically all of {x_c, x_d, u, p, t} should be pulled from υ
    # here, we define all connections identity for {x_d, p, u, t} and pull only {x_c}
    # u=[] in this example
    u = υ_b
    mlm.g(γ_b, HUDADE.NoVector, HUDADE.NoVector, u, p, t)
    nothing
end

cfct = CombinedEquationSystem(f_a, f_b; 
    len_u=length(x_c0), 
    len_u_a=length(x_c0),  
    len_u_b=length(x_c0), 
    len_y=length(x_c0),  
    len_y_a=length(x_c0), 
    len_y_b=length(x_c0))

hm_f = function(ẋ_c, x_c, x_d, u, p, t)
    eval!(cfct, ẋ_c, x_c, x_d, u, p, t)

    nothing
end

hm_α = function(x_c, x_d, u, p, t)
    fpm_p = p0

    fpm.α(x_c, x_d, u, fpm_p, t)

    x = solution_gt[1](t)
    x_c[:] = x[1:4]
    x_d[1] = x[5]
    #x_d[2] = 2.0
    nothing
end

hm_ω = function()
    fpm.ω()
    nothing
end

hm_c_x = function(z, x_c, x_d, u, p)
    fpm_p = p0
    
    fpm.c_x(z, x_c, x_d, u, fpm_p)
    nothing
end

hm_a_x = function(x_c_right_local, x_d_right_local, x_c_left, x_d_left, u, p, t, idx)
    fpm_p = p0
    
    fpm.a_x(x_c_right_local, x_d_right_local, x_c_left, x_d_left, u, fpm_p, t, idx)
    nothing
end

hm_g = function(y, x_c, x_d, u, p, t)
    nothing
end

hm = HUDAODEFunction(; z_len=fpm.z_len, x_c_len=length(x_c0), x_d_len=length(x_d0), f=hm_f, α=hm_α, ω=hm_ω, c_x=hm_c_x, a_x=hm_a_x, g=hm_g)

global horizon = round(Int, length(ts)/10)
global problem = nothing

scale_s1 = 0.5 # -1.0 / (min(data_s1...) - max(data_s1...))
scale_v1 = 0.1 # -1.0 / (min(data_v1...) - max(data_v1...))
scale_s2 = scale_s1
scale_v2 = scale_v1

# function min_max_reg(A; factor=1.0)
#     _min = min(A...)
#     _max = max(A...)

#     loss = 0.0

#     loss += abs(0.0 - _min)*factor
#     loss += abs(1.0 - _max)*factor
    
#     return loss
# end

# function pos_reg(A; factor=1.0)
#     _min = min(A...)
   
#     loss = 0.0

#     if _min < 0.0
#         loss += (0.0 - _min)*factor
#     end
    
#     return loss
# end

# order = 4
# k0_est = [16.0, -32.0, 16.0, 0.0, 0.0]
# function poly(x, k=k0_est; order=order)
#     y = 0.0 
#     for i in 0:order 
#         y += k[i+1] * x^(order-i)
#     end
#     return y 
# end

# function poly_reg(A; factor=0.01)
    
#     loss = 0.0
#     num = length(A)

#     for i in 1:num
#         loss += poly(A[i])/num*factor
#     end
    
#     return loss
# end

function loss(p; include_reg::Bool=true, i = rand(1:numSims))

    iStart = 1
    iStop = horizon 
    tspan = (t0, t0 + (horizon-1)*dt)
    ts = tspan[1]:dt:tspan[end]

    data_s1 = collect(u[1] for u in solution_gt[i].u)
    data_v1 = collect(u[2] for u in solution_gt[i].u)
    data_s2 = collect(u[3] for u in solution_gt[i].u)
    data_v2 = collect(u[4] for u in solution_gt[i].u)

    sol = solve(problem, Tsit5(); tspan=tspan, u0=x0[i], saveat=ts, p=p) # , sensealg=SciMLSensitivity.ReverseDiffAdjoint())

    # fig = plot(data1, data3; title="$(tspan)", label="Data")
    # plot!(fig, collect(u[1] for u in sol.u), collect(u[3] for u in sol.u); label="Sol")
    # display(fig)
    
    s1 = nothing 
    v1 = nothing 
    s2 = nothing
    v2 = nothing
    if isa(sol, ReverseDiff.TrackedArray)
        s1 = sol[1,:] 
        v1 = sol[2,:]
        s2 = sol[3,:] 
        v2 = sol[4,:]
    else
        s1 = collect(u[1] for u in sol.u)
        v1 = collect(u[2] for u in sol.u)
        s2 = collect(u[3] for u in sol.u)
        v2 = collect(u[4] for u in sol.u)
    end

    loss = 0.0

    if include_reg
        # if hasproperty(p, :W_az)
        #     loss += poly_reg(p[:W_az][:values])
        # end
        # if hasproperty(p, :W_ba)
        #     loss += poly_reg(p[:W_ba][:values])
        # end
        # if hasproperty(p, :W_bz)
        #     loss += poly_reg(p[:W_bz][:values])
        # end
        # if hasproperty(p, :W_za)
        #     loss += poly_reg(p[:W_za][:values])
        # end
        # if hasproperty(p, :W_zb)
        #     loss += poly_reg(p[:W_zb][:values])
        # end
        # if hasproperty(p, :W_zz)
        #     loss += poly_reg(p[:W_zz][:values])
        # end
    end
    
    loss += sum(abs.(s1 .- data_s1[iStart:iStop])) * scale_s1 / horizon + 
        sum(abs.(s2 .- data_s2[iStart:iStop])) * scale_s2 / horizon +
        sum(abs.(v1 .- data_v1[iStart:iStop])) * scale_v1 / horizon +
        sum(abs.(v2 .- data_v2[iStart:iStop])) * scale_v2 / horizon 

    return loss
end

global u_latest = nothing
global u_pre_latest
function callback(state, val, topologyStr)
    global u_latest, u_pre_latest
    global horizon, meter, iteration

    wo_val = 0.0 
    losses = []

    for i in 1:numSims
        l = loss(state; i=i)
        wo_val = max(wo_val, l)
        push!(losses, l)
    end
    
    if wo_val < 0.05 && horizon < length(ts) # 0.005
        horizon += 1
    end

    iteration += 1
    pro = iteration/numIterations
    # if pro > 1.0
    #     pro = 1.0
    # end
    # meter.desc = "$(topologyStr) | $(round(horizon*100/length(ts); digits=1))% | $(round(val; digits=6)) |"
    # ProgressMeter.update!(meter, floor(Integer, 1000.0*pro))
    if iteration % 10 == 0
        reg_per = (val - wo_val) / val * 100.0
        lstr = ""
        for l in losses 
            lstr *= "\n$(round(l; digits=6))"
        end
        @info "$(topologyStr) | H:$(round(horizon*100/length(ts); digits=1))% | $(round(pro*100; digits=1))%" * lstr
    end

    u_pre_latest = u_latest

    if VERSION >= v"1.10"
        u_latest = copy(state.u)
    else
        u_latest = copy(state)
    end

    return false
end

# test loop 
parallels = (true, false)
sequentials = (true, false)
feedthroughs = (true, false)
optimParams = (true, ) 

# global meter = ProgressMeter.Progress(1000; desc="...", color=:blue, dt=1.0)
global iteration = 0
global numIterations = 20000
solutions = []
#ProgressMeter.update!(meter, 0)
for parallel in parallels
    for sequential in sequentials
        for feedthrough in feedthroughs
            for optimParam ∈ optimParams

# parallel = true 
# sequential = false
# feedthrough = false
# optimParam = false

                cfct.parallel = parallel
                cfct.sequential = sequential
                cfct.feedthrough = feedthrough

                topologyStr = "P$(parallel ? 1 : 0)S$(sequential ? 1 : 0)F$(feedthrough ? 1 : 0)O$(optimParam ? 1 : 0)"

                global problem
                global horizon, meter, iteration, numIterations
                
                iteration = 0
                horizon = round(Int, length(ts)/20)
                #meter.desc = "$(topologyStr) | 0% |"
                #ProgressMeter.update!(meter, 0)

                nt = getParameters(cfct; train=optimParam, bias=true)

                hm_p0 = nothing
                if optimParam
                    hm_p0 = ComponentArray(merge(nt, deepcopy(mlm_p0)))
                else
                    hm_p0 = deepcopy(mlm_p0)
                end

                # HM Test
                problem = HUDAODEProblem(hm, x_c0, x_d0, tspan; p=hm_p0)
                solution = solve(problem)
                fig = plotBB(collect(solution(t)[1] for t in ts), collect(solution(t)[3] for t in ts), title="$(topologyStr)")
                display(fig)

                adtype = Optimization.AutoForwardDiff(;chunksize=32)
                #adtype = Optimization.AutoReverseDiff()

                optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
                optprob = Optimization.OptimizationProblem(optf, hm_p0)

                loss_before = loss(optprob.u0)

                # result = Optimization.solve(optprob, OptimizationOptimisers.Adam(1e-2); callback=(state, val) -> callback(state, val, topologyStr), maxiters=round(Integer, numIterations/2))
                # optprob = remake(optprob; u0=u_pre_latest)

                # loss(optprob.u0)
                
                result = Optimization.solve(optprob, OptimizationOptimisers.Adam(1e-3); callback=(state, val) -> callback(state, val, topologyStr), maxiters=numIterations)
                optprob = remake(optprob; u0=u_pre_latest)

                loss_after = loss(optprob.u0)

                fig = plot(layout=grid(numSims,1), size=(300,numSims*300))
                for i in 1:numSims
                    sol = solve(problem, Tsit5(); p=optprob.u0, u0=x0[i], saveat=ts)
                    plot!(fig[i], collect(u[1] for u in solution_fpm[i].u), collect(u[3] for u in solution_fpm[i].u), style=:dot, label="FPM", title="$(topologyStr)")
                    plot!(fig[i], collect(u[1] for u in solution_gt[i].u[1:horizon]), collect(u[3] for u in solution_gt[i].u[1:horizon]), label="GT")
                    plot!(fig[i], collect(u[1] for u in solution_gt[i].u[horizon:end]), collect(u[3] for u in solution_gt[i].u[horizon:end]), style=:dash, label=:none)
                    plot!(fig[i], collect(u[1] for u in sol.u)[1:horizon], collect(u[3] for u in sol.u)[1:horizon], label="HM")
                    plot!(fig[i], collect(u[1] for u in sol.u)[horizon:end], collect(u[3] for u in sol.u)[horizon:end], style=:dash, label=:none)
                end
                display(fig)
                savefig(fig, joinpath(@__DIR__, "plots", "png", "$(topologyStr).png"))
                savefig(fig, joinpath(@__DIR__, "plots", "svg", "$(topologyStr).svg"))

                sol = solve(problem, Tsit5(); p=optprob.u0, saveat=ts)
                sol_test = solve(problem, Tsit5(); p=optprob.u0, u0=x0_test, saveat=ts)

                horizon = length(ts)
                loss_full = loss(optprob.u0)
                push!(solutions, (sol, sol_test, topologyStr, loss_full, deepcopy(optprob.u0)))

                xticks1 = (1:4, [L"\dot{x}_{a1}", L"\dot{x}_{a2}", L"\dot{x}_{a3}", L"\dot{x}_{a4}"])
                xticks2 = (1:4, [L"\dot{x}_{b1}", L"\dot{x}_{b2}", L"\dot{x}_{b3}", L"\dot{x}_{b4}"])
                xticks3 = (1:4, [L"x_{z1}", L"x_{z2}", L"x_{z3}", L"x_{z4}"])

                yticks1 = (1:4, [L"x_{a1}", L"x_{a2}", L"x_{a3}", L"x_{a4}"])
                yticks2 = (1:4, [L"x_{b1}", L"x_{b2}", L"x_{b3}", L"x_{b4}"])
                yticks3 = (1:4, [L"\dot{x}_{z1}", L"\dot{x}_{z2}", L"\dot{x}_{z3}", L"\dot{x}_{z4}"])

                textsize = 16
                plotkwargs = Dict(:xtickfontsize=>textsize, :ytickfontsize=>textsize, :titlesize=>textsize)
                fig = plot(; layout=grid(1,6), size=(1200,200+13), dpi=300, bottom_margin=10mm, top_margin=3mm, plotkwargs...) # , plot_title="$(topologyStr)", plot_titlevspan=0.2)

                plotMatrix!(fig[1], (hasproperty(optprob.u0, :W_az) ? reshape(optprob.u0[:W_az][:values], 4, 4) : cfct.W_az), prefix="\\mathbf{W}_{az} ", 
                    xticks=xticks3, yticks=yticks1 )
                plotMatrix!(fig[2], (hasproperty(optprob.u0, :W_ba) ? reshape(optprob.u0[:W_ba][:values], 4, 4) : cfct.W_ba), prefix="\\mathbf{W}_{ba} ", 
                    xticks=xticks1, yticks=yticks2 )
                plotMatrix!(fig[3], (hasproperty(optprob.u0, :W_bz) ? reshape(optprob.u0[:W_bz][:values], 4, 4) : cfct.W_bz), prefix="\\mathbf{W}_{bz} ", 
                    xticks=xticks3, yticks=yticks2 )

                plotMatrix!(fig[4], (hasproperty(optprob.u0, :W_za) ? reshape(optprob.u0[:W_za][:values], 4, 4) : cfct.W_za), prefix="\\mathbf{W}_{za} ",  
                    xticks=xticks1, yticks=yticks3 ) # how much FPM?
                plotMatrix!(fig[5], (hasproperty(optprob.u0, :W_zb) ? reshape(optprob.u0[:W_zb][:values], 4, 4) : cfct.W_zb), prefix="\\mathbf{W}_{zb} ",  
                    xticks=xticks2, yticks=yticks3 ) # how much MLM?
                plotMatrix!(fig[6], (hasproperty(optprob.u0, :W_zz) ? reshape(optprob.u0[:W_zz][:values], 4, 4) : cfct.W_zz), prefix="\\mathbf{W}_{zz} ",  
                    xticks=xticks3, yticks=yticks3 ) # how much DFT?

                display(fig)
                savefig(fig, joinpath(@__DIR__, "plots", "png", "$(topologyStr)_W.png"))
                savefig(fig, joinpath(@__DIR__, "plots", "svg", "$(topologyStr)_W.svg"))
            end
        end
    end
end

colors = Colors.distinguishable_colors(6, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

for i in 1:numSims
    fig = plotBB(; legend=:topright)
    plot!(fig, collect(u[1] for u in solution_gt[i].u), collect(u[3] for u in solution_gt[i].u), label="ground truth", color=:black)
    plot!(fig, collect(u[1] for u in solution_fpm[i].u), collect(u[3] for u in solution_fpm[i].u), label="FPM", style=:dash, color=:black)
    c = 1
    for solution in solutions
        sol, sol_test, topologyStr, loss_full, p = solution 

        nt = getParameters(cfct; train=true)
        hm_p0 = ComponentArray(merge(nt, p))
        problem = HUDAODEProblem(hm, x_c0, x_d0, tspan; p=hm_p0)
        sol = solve(problem, Tsit5(); p=p, u0=x0[i], saveat=ts)

        if loss_full < 0.2
            println(shortString(topologyStr))

            label = "$(shortString(topologyStr))" #  [MAE: $(round(loss_full*1000; digits=2))e-3]
            plot!(fig, collect(u[1] for u in sol.u), collect(u[3] for u in sol.u), label=label, color=colors[c])
            c += 1
        else
            println("[ERROR] $(round(loss_full; digits=3)) -> $(shortString(topologyStr))")
        end
    end
    scatterlabels = collect(e.t for e in solution_gt[i].events)
    scatterlabels = [ts[1], scatterlabels..., ts[end]] # [[1,2,3,5,6,8,9,11,12,13]]
    scatterBB!(fig, solution_gt[i]; scatterlabels=scatterlabels)

    savefig(fig, joinpath(@__DIR__, "plots", "png", "topologies_$(i).png"))
    savefig(fig, joinpath(@__DIR__, "plots", "svg", "topologies_$(i).svg"))
    display(fig) 
end

#####

tspan_test = (0.0, 2.1)
ts_test = tspan_test[1]:dt:tspan_test[end]
x0_test2 = copy(x0[3])
x0_test2[1] = 0.75 # -x0_test2[1]
x0_test2[3] = 0.25 # -x0_test2[3]

solution_test = solve(problem_gt; saveat=ts_test, u0=x0_test2)

fig = plotBB(; legend=:topleft)
plot!(fig, collect(u[1] for u in solution_test.u), collect(u[3] for u in solution_test.u), label="ground truth", color=:black)
#plot!(fig, collect(u[1] for u in solution_fpm.u), collect(u[3] for u in solution_fpm.u), label="FPM", style=:dash, color=:black)
i = 1
c = 1
for solution in solutions[1:2]
    sol, sol_test, topologyStr, loss_full, p = solution
    
    nt = getParameters(cfct; train=true)
    hm_p0 = ComponentArray(merge(nt, p))
    problem = HUDAODEProblem(hm, x_c0, x_d0, tspan_test; p=hm_p0)
    sol_test = solve(problem, Tsit5(); p=p, u0=x0_test2, saveat=ts_test)

    if loss_full < 0.2
        println(shortString(topologyStr))

        label = "$(shortString(topologyStr))" #  [MAE: $(round(loss_full*1000; digits=2))e-3]
        plot!(fig, collect(u[1] for u in sol_test.u), collect(u[3] for u in sol_test.u), label=label, color=colors[i])
        i += 1
    else
        println("[ERROR] $(round(loss_full; digits=3)) -> $(shortString(topologyStr))")
    end
end
scatterlabels = collect(e.t for e in solution_test.events)
scatterlabels = [ts_test[1], scatterlabels..., ts_test[end]]
scatterBB!(fig, solution_test; scatterlabels=scatterlabels)
fig 

savefig(fig, joinpath(@__DIR__, "plots", "png", "topologies_extra.png"))
savefig(fig, joinpath(@__DIR__, "plots", "svg", "topologies_extra.svg"))
display(fig)

### correlation test

using Statistics

function cor_coef(j, i, sol=solution_gt[1]; len=length(ts))
    a = collect(u[i] for u in sol.u)[1:len]
    b = collect(sol(t, Val{1})[j] for t in sol.t)[1:len]

    for i in 1:length(b)
        if abs(b[i]) > 100 
            b[i] = 0.0
        end
    end

    std_a = std(a)
    mean_a = mean(a)
    std_b = std(b)
    mean_b = mean(b)
    len = length(a)

    kov = sum( (a .- mean_a) .* (b .- mean_b) ) / (len - 1)  

    kov_kof = kov / (std_a * std_b)
    return kov_kof 
end

A = zeros(4,4)
for i in 1:4 
    for j in 1:4
        A[i,j] = cor_coef(i,j)
    end
end
plot(Gray.(abs.(A)))

fig = plotCC() # plotCC([10, 10], [-1, 1], style=:dash, label="Start Horizon", color=:black)
plot!(fig, [h[1], h[end]], [0,0]; label=:none, style=:dash, color=:black)
colors = Colors.distinguishable_colors(6, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
c=1
for s in (1,2,3)
    #W_zz_43 = []
    W_zz_23 = []
    #W_zz_12 = []
    #W_zz_22 = []
    W_zz_21 = []
    for i in 1:length(ts)
        #push!(W_zz_43, cor_coef(4,3; len=i))
        push!(W_zz_23, cor_coef(2,3, solution_gt[s]; len=i))
        #push!(W_zz_12, cor_coef(1,2; len=i))
        #push!(W_zz_22, cor_coef(2,2; len=i))
        push!(W_zz_21, cor_coef(2,1, solution_gt[s]; len=i))
    end

    h = collect(1:length(ts)) ./ length(ts) .* 100

    plot!(fig, h, W_zz_21; label=L"W_{zz}[2,1]" * " (S$(s))", color=colors[c])
    #plot!(fig, [h[1], h[end]], [0, 0]; label=:none, style=:dash, color=colors[i])
    c += 1
    plot!(fig, h, W_zz_23; label=L"W_{zz}[2,3]" * " (S$(s))", color=colors[c])
    #plot!(fig, [h[1], h[end]], [1, 1]; label=:none, style=:dash, color=colors[i])
    c += 1
end
display(fig)
savefig(fig, joinpath(@__DIR__, "plots", "png", "cc.png"))
savefig(fig, joinpath(@__DIR__, "plots", "svg", "cc.svg"))

function poly_der(x, k; order=7)
    y = 0.0 
    for i in 0:(order-1) 
        y += (order-i) * k[i+1] * x^(order-i-1)
    end
    return y 
end

poly(2.0, [4,2,3]; order=2)

points = [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]]
points_pos = [[-0.1, 1.0], [1.1, 1.0]]
function poly_loss(k; order=7)

    e = 0.0
    for point in points 
        x, y = point 
        ŷ = poly(x, k; order=order)
        der = poly_der(x, k; order=order)
        e += abs(ŷ - y) + abs(der - 0.0)
    end
    # for point in points_pos
    #     x, y = point 
    #     ŷ =  poly(x, k; order=order)
    #     if ŷ < 0.0
    #         e -= ŷ
    #     end
    # end
    return e
end

using ForwardDiff

order = 4
k0 = zeros(order+1)
poly_loss(k0; order=order)

step = 1e-2
for i in 1:100000
    grad = ForwardDiff.gradient(k -> poly_loss(k; order=order), k0)
    k0 -= grad*step
end

poly_loss(k0; order=order)

step = 1e-3
for i in 1:100000
    grad = ForwardDiff.gradient(k -> poly_loss(k; order=order), k0)
    k0 -= grad*step
end

poly_loss(k0; order=order)

step = 1e-4
for i in 1:100000
    grad = ForwardDiff.gradient(k -> poly_loss(k; order=order), k0)
    k0 -= grad*step
end

poly_loss(k0; order=order)

plot(ts, collect(poly(x, k0; order=order) for x in ts), ylims=(-0.5, 1.5))
scatter!(collect(x[1] for x in points), collect(x[2] for x in points))

poly_loss(k0_est; order=order)
plot!(ts, collect(poly(x, k0_est; order=order) for x in ts), ylims=(-0.5, 1.5))