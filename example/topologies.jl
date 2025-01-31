#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

include(joinpath(@__DIR__, "helpers.jl"))

import HUDADE: BouncingBall2D, BouncingBall2D_p0

fpm_p0_fric = copy(BouncingBall2D_p0)
fpm_p0_fric[5] = 0.15
fpm_p0_wo_fric = copy(fpm_p0_fric)
fpm_p0_wo_fric[5] = 0.0

u0 = zeros(Float64, 0)
t0 = 0.0
z0 = zeros(4)
x_c0 = [-0.25, -6.0, 0.5, 8.0]
x_d0 = [0.0] # , 2.0]
tspan = (t0, 2.1)
dt = 0.01
ts = tspan[1]:dt:tspan[2]

x0 = [[-0.25, -6.0, 0.5, 8.0, 0.0], [0.5, 8.0, -0.25, -6.0, 0.0], [0.0, 0.0, 0.0, 4.0, 0.0], [-0.75, 4.0, -0.5, 6.0, 0.0], [-0.5, 2.0, 0.5, 2.0, 0.0]]

numSims = 4
numTests = 1
solution_gt = Vector{HUDADESolution}(undef, numSims+numTests)
solution_fpm = Vector{HUDADESolution}(undef, numSims+numTests)

# FPM Test
gt = BouncingBall2D(; x_c0=x_c0, x_d0=x_d0, p=fpm_p0_fric, tspan=tspan);
for i in 1:numSims+numTests
    solution_gt[i] = solve(gt; x0=x0[i], saveat=ts)

    fig = plot(collect(u[1] for u in solution_gt[i].u), collect(u[3] for u in solution_gt[i].u))
    display(fig)
end

# scatter(collect(solution_gt[1](t, Val{1})[2] for t in solution_gt[1].t), 
#     collect(solution_gt[1](t, Val{1})[4] for t in solution_gt[1].t) .+ 9.81,
#     xlims=(-15,15), ylims=(-15,15))

# cor_coef(2, 4, solution_gt[5])
# cor_coef(4, 2, solution_gt[4])

# start with a fresh one!
fpm = BouncingBall2D(; x_c0=x_c0, x_d0=x_d0, p=fpm_p0_wo_fric, tspan=tspan);
for i in 1:numSims+numTests
    solution_fpm[i] = solve(fpm; x0=x0[i], saveat=ts)

    fig = plot(collect(u[1] for u in solution_fpm[i].u), collect(u[3] for u in solution_fpm[i].u))
    display(fig)
end

# MLM Test
ann = Chain(Dense(4, 8, tanh),
            Dense(8, 2, tanh))
            #Dense(8, 2, identity)) 
mlm = HUDADE.LuxSecondOrderModel(ann)
mlm_p0 = mlm.p
solution_mlm = solve(mlm; saveat=ts)

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
    fpm_p = fpm_p0_wo_fric
    fpm.fct.f(γ_a, x_c, x_d, u, fpm_p, t)
    nothing
end

function f_b(γ_b::AbstractArray{<:Real}, υ_b::AbstractArray{<:Real}, x_d, u, p, t)
    # basically all of {x_c, x_d, u, p, t} should be pulled from υ
    # here, we define all connections identity for {x_d, p, u, t} and pull only {x_c}
    # u=[] in this example
    u = υ_b
    mlm.fct.g(γ_b, HUDADE.NoVector, HUDADE.NoVector, u, p, t)
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
    fpm_p = fpm_p0_wo_fric

    fpm.fct.α(x_c, x_d, u, fpm_p, t)

    x = solution_gt[1](t)
    x_c[:] = x[1:4]
    x_d[1] = x[5]
    #x_d[2] = 2.0
    nothing
end

hm_ω = function()
    fpm.fct.ω()
    nothing
end

hm_c_x = function(z, x_c, x_d, u, p)
    fpm_p = fpm_p0_wo_fric
    
    fpm.fct.c_x(z, x_c, x_d, u, fpm_p)
    nothing
end

hm_a_x = function(x_c_right_local, x_d_right_local, x_c_left, x_d_left, u, p, t, idx)
    fpm_p = fpm_p0_wo_fric
    
    fpm.fct.a_x(x_c_right_local, x_d_right_local, x_c_left, x_d_left, u, fpm_p, t, idx)
    nothing
end

hm_g = function(y, x_c, x_d, u, p, t)
    nothing
end

# z_len=fpm.z_len, 
hm = HUDAODEFunction(; 
    z_len=fpm.fct.z_len, 
    x_c_len=length(x_c0), 
    x_d_len=length(x_d0), 
    f=hm_f, 
    α=hm_α, 
    ω=hm_ω, 
    c_x=hm_c_x, 
    a_x=hm_a_x, 
    g=hm_g)

global horizon = round(Int, length(ts)/10)
global problem = nothing

scale_s1 = 0.5 # -1.0 / (min(data_s1...) - max(data_s1...))
scale_v1 = 0.1 # -1.0 / (min(data_v1...) - max(data_v1...))
scale_s2 = scale_s1
scale_v2 = scale_v1

# function poly_reg(A; factor=0.01)
    
#     loss = 0.0
#     num = length(A)

#     for i in 1:num
#         loss += poly(A[i])/num*factor
#     end
    
#     return loss
# end

function loss(p; include_reg::Bool=true, i=rand(1:numSims), horizon=horizon)

    iStart = 1
    iStop = horizon 
    tspan = (t0, t0 + (horizon-1)*dt)
    ts = tspan[1]:dt:tspan[end]

    data_s1 = collect(u[1] for u in solution_gt[i].u)
    data_v1 = collect(u[2] for u in solution_gt[i].u)
    data_s2 = collect(u[3] for u in solution_gt[i].u)
    data_v2 = collect(u[4] for u in solution_gt[i].u)

    sol = solve(problem, Tsit5(); tspan=tspan, x0=x0[i], saveat=ts, p=p) # , sensealg=SciMLSensitivity.ReverseDiffAdjoint())

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

    if VERSION >= v"1.10"
        state = state.u
    end

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

    Optimisers.adjust!(state.original, 1e-3 - (1.0-pro)*(1e-3 - 1e-4) )
    # if pro > 1.0
    #     pro = 1.0
    # end
    # meter.desc = "$(topologyStr) | $(round(horizon*100/length(ts); digits=1))% | $(round(val; digits=6)) |"
    # ProgressMeter.update!(meter, floor(Integer, 1000.0*pro))
    if iteration % min(numIterations/2, 500) == 0
        reg_per = (val - wo_val) / val * 100.0
        lstr = ""
        for l in losses 
            lstr *= "\n$(round(l; digits=6))"
        end
        @info "$(topologyStr) | H:$(round(horizon*100/length(ts); digits=1))% | $(round(pro*100; digits=1))%" * lstr
    end

    u_pre_latest = u_latest
    u_latest = copy(state)
    
    return false
end

# test loop 
parallels = (true, false)
sequentials = (true, false)
feedthroughs = (true, false)
biases = (false, )
optimParams = (true, ) 

xticks1 = (1:4, [L"\dot{x}_{1,a}", L"\dot{x}_{2,a}", L"\dot{x}_{3,a}", L"\dot{x}_{4,a}"])
xticks2 = (1:4, [L"\dot{x}_{1,b}", L"\dot{x}_{2,b}", L"\dot{x}_{3,b}", L"\dot{x}_{4,b}"])
xticks3 = (1:4, [L"x_{1,z}", L"x_{2,z}", L"x_{3,z}", L"x_{4,z}"])
# xticks1 = (1:4, [L"\dot{s}_{x}^{(a)}", L"\dot{v}_{x}^{(a)}", L"\dot{s}_{y}^{(a)}", L"\dot{v}_{y}^{(a)}"])
# xticks2 = (1:4, [L"\dot{x}_{1}^{b}", L"\dot{x}_{2}^{b}", L"\dot{x}_{3}^{b}", L"\dot{x}_{1}^{b}"])
# xticks3 = (1:4, [L"s_x", L"v_x", L"s_y", L"v_y"])

yticks1 = (1:4, [L"x_{1,a}", L"x_{2,a}", L"x_{3,a}", L"x_{4,a}"])
yticks2 = (1:4, [L"x_{1,b}", L"x_{2,b}", L"x_{3,b}", L"x_{4,b}"])
yticks3 = (1:4, [L"\dot{x}_{1,z}", L"\dot{x}_{2,z}", L"\dot{x}_{3,z}", L"\dot{x}_{4,z}"])
# yticks1 = (1:4, [L"s_{x}^{(a)}", L"v_{x}^{(a)}", L"s_{y}^{(a)}", L"y_{y}^{(a)}"])
# yticks2 = (1:4, [L"{}^{b}s_{x}", L"{}^{b}v_{x}", L"{}^{b}s_{y}", L"{}^{b}y_{y}"])
# yticks3 = (1:4, [L"\dot{s}_{x}", L"\dot{v}_{x}", L"\dot{s}_{y}", L"\dot{v}_{y}"])

# global meter = ProgressMeter.Progress(1000; desc="...", color=:blue, dt=1.0)
global iteration = 0
global numIterations = 20000
solutions = []
#ProgressMeter.update!(meter, 0)
for parallel in parallels
    for sequential in sequentials
        for feedthrough in feedthroughs
            for bias in biases
                for optimParam ∈ optimParams

                    # boring
                    if !parallel && !sequential && !feedthrough
                        continue
                    end

                    # instable
                    if !parallel && !sequential && feedthrough
                        continue
                    end

# parallel = true 
# sequential = false
# feedthrough = false
# optimParam = false

                    cfct.parallel = parallel
                    cfct.sequential = sequential
                    cfct.feedthrough = feedthrough

                    topologyStr = "P$(parallel ? 1 : 0)S$(sequential ? 1 : 0)D$(feedthrough ? 1 : 0)O$(optimParam ? 1 : 0)"

                    global problem
                    global horizon, meter, iteration, numIterations
                    
                    iteration = 0
                    horizon = round(Int, length(ts)/20)
                    #meter.desc = "$(topologyStr) | 0% |"
                    #ProgressMeter.update!(meter, 0)

                    nt = getParameters(cfct; initialize=optimParam, connect=false, bias=bias)

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

                    fig = plot(layout=grid(numSims+numTests,1), size=(300,(numSims+numTests)*300), dpi=300, xlims=(-1,1), ylims=(-1,1))
                    sols = Vector{HUDADESolution}(undef, numSims+numTests)
                    losses = Vector{Float64}(undef, numSims+numTests)
                    for i in 1:numSims+numTests
                        sols[i] = solve(problem, Tsit5(); p=optprob.u0, x0=x0[i], saveat=ts)
                        losses[i] = loss(optprob.u0; i=i, horizon=length(ts))
                        plot!(fig[i], collect(u[1] for u in solution_fpm[i].u), collect(u[3] for u in solution_fpm[i].u), style=:dot, label="FPM", title="$(topologyStr)")
                        plot!(fig[i], collect(u[1] for u in solution_gt[i].u[1:horizon]), collect(u[3] for u in solution_gt[i].u[1:horizon]), label="GT")
                        plot!(fig[i], collect(u[1] for u in solution_gt[i].u[horizon:end]), collect(u[3] for u in solution_gt[i].u[horizon:end]), style=:dash, label=:none)
                        plot!(fig[i], collect(u[1] for u in sols[i].u)[1:horizon], collect(u[3] for u in sols[i].u)[1:horizon], label="HM")
                        plot!(fig[i], collect(u[1] for u in sols[i].u)[horizon:end], collect(u[3] for u in sols[i].u)[horizon:end], style=:dash, label=:none)
                    end
                    display(fig)
                    savefig(fig, joinpath(@__DIR__, "plots", "png", "$(topologyStr).png"))
                    savefig(fig, joinpath(@__DIR__, "plots", "pdf", "$(topologyStr).pdf"))

                    #sol = solve(problem, Tsit5(); p=optprob.u0, saveat=ts)
                    
                    horizon = length(ts)
                    loss_full = loss(optprob.u0)
                    push!(solutions, (sols, losses, topologyStr, loss_full, deepcopy(optprob.u0)))

                    textsize = 16
                    plotkwargs = Dict(:xtickfontsize=>textsize, :ytickfontsize=>textsize, :titlesize=>textsize)

                    layouts = [(1,6), (2,3)]
                    sizes = [(1200,200+9), (600,400+125)]
                    for j in 1:length(layouts)
                        layout = layouts[j]
                        size = sizes[j]
                        fig = plot(; layout=grid(layout[1], layout[2]), size=size, dpi=300, bottom_margin=10mm, top_margin=3mm, plotkwargs...) # , plot_title="$(topologyStr)", plot_titlevspan=0.2)

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

                        layoutStr = replace("$(layout)", "("=>"", ")"=>"", ","=>"", " "=>"")
                        savefig(fig, joinpath(@__DIR__, "plots", "png", "$(topologyStr)_W_$(layoutStr).png"))
                        savefig(fig, joinpath(@__DIR__, "plots", "pdf", "$(topologyStr)_W_$(layoutStr).pdf"))
                    end # for layouts

                end
            end
        end
    end
end

colors = Colors.distinguishable_colors(7, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

for i in 1:numSims+numTests
    fig = plotBB(; ) # legend=(i <= numSims ? :topright : :topleft))
    plot!(fig, collect(u[1] for u in solution_gt[i].u), collect(u[3] for u in solution_gt[i].u), label="ground truth", color=:black)
    plot!(fig, collect(u[1] for u in solution_fpm[i].u), collect(u[3] for u in solution_fpm[i].u), label="FPM", style=:dash, color=:black)
    c = 1
    for solution in solutions
        sols, losses, topologyStr, loss_full, p = solution 

        sol = sols[i]
        l = losses[i]

        # nt = getParameters(cfct; initialize=true, connect=false, bias=false)
        # hm_p0 = p # ComponentArray(merge(nt, p))
        # problem = HUDAODEProblem(hm, x_c0, x_d0, tspan; p=hm_p0)
        # sol = solve(problem, Tsit5(); p=hm_p0, x0=x0[i], saveat=ts)

        if true # loss_full < 0.05
            println(shortString(topologyStr))

            label = "$(shortString(topologyStr)) [$(round(l; digits=3))]" #  [MAE: $(round(loss_full*1000; digits=2))e-3]
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
    savefig(fig, joinpath(@__DIR__, "plots", "pdf", "topologies_$(i).pdf"))
    display(fig) 
end

# extrapolation focused
for i in 5:5
    fig = plotBB(; legend=(i <= numSims ? :topright : :topright))
    plot!(fig, collect(u[1] for u in solution_gt[i].u), collect(u[3] for u in solution_gt[i].u), label="ground truth", color=:black)
    plot!(fig, collect(u[1] for u in solution_fpm[i].u), collect(u[3] for u in solution_fpm[i].u), label="FPM", style=:dash, color=:black)
    c = 1
    for solution in solutions
        sols, losses, topologyStr, loss_full, p = solution 

        sol = sols[i]
        l = losses[i]

        # nt = getParameters(cfct; initialize=true, connect=false, bias=false)
        # hm_p0 = p # ComponentArray(merge(nt, p))
        # problem = HUDAODEProblem(hm, x_c0, x_d0, tspan; p=hm_p0)
        # sol = solve(problem, Tsit5(); p=hm_p0, x0=x0[i], saveat=ts)

        if true # loss_full < 0.05
            println(shortString(topologyStr))

            label = "$(shortString(topologyStr)) [$(round(l; digits=3))]" #  [MAE: $(round(loss_full*1000; digits=2))e-3]
            plot!(fig, collect(u[1] for u in sol.u), collect(u[3] for u in sol.u), label=label, color=colors[c])
            c += 1
        else
            println("[ERROR] $(round(loss_full; digits=3)) -> $(shortString(topologyStr))")
        end
    end
    scatterlabels = collect(e.t for e in solution_gt[i].events)
    scatterlabels = [ts[1], scatterlabels..., ts[end]] # [[1,2,3,5,6,8,9,11,12,13]]
    scatterBB!(fig, solution_gt[i]; scatterlabels=scatterlabels)

    savefig(fig, joinpath(@__DIR__, "plots", "png", "topologies_extra.png"))
    savefig(fig, joinpath(@__DIR__, "plots", "pdf", "topologies_extra.pdf"))
    display(fig) 
end

### correlation test

using Statistics

function cor_coef(a_index, b_index, sol=solution_gt[1]; a_deriv=1, b_deriv=1, len=length(ts))
    a = collect(sol(t, Val{a_deriv})[a_index] for t in sol.t)[1:len]
    b = collect(sol(t, Val{b_deriv})[b_index] for t in sol.t)[1:len]

    if a_deriv >= 1
        for i in 1:length(a)
            if abs(a[i]) > 100 
                a[i] = 0.0
            end
        end
    end

    if b_deriv >= 1
        for i in 1:length(b)
            if abs(b[i]) > 100 
                b[i] = 0.0
            end
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
h = collect(1:length(ts)) ./ length(ts) .* 100
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

    plot!(fig, h, W_zz_21; label=L"W_{zz}[2,1]" * " (S$(s))", color=colors[c])
    #plot!(fig, [h[1], h[end]], [0, 0]; label=:none, style=:dash, color=colors[i])
    c += 1
    plot!(fig, h, W_zz_23; label=L"W_{zz}[2,3]" * " (S$(s))", color=colors[c])
    #plot!(fig, [h[1], h[end]], [1, 1]; label=:none, style=:dash, color=colors[i])
    c += 1
end
display(fig)
savefig(fig, joinpath(@__DIR__, "plots", "png", "cc.png"))
savefig(fig, joinpath(@__DIR__, "plots", "pdf", "cc.pdf"))