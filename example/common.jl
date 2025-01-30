#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

include(joinpath(@__DIR__, "helpers.jl"))

# variables
x_c0 = [-0.25, -6.0, 0.5, 8.0]
x_d0 = [0.0]
x0 = [x_c0..., x_d0...]
t0 = 0.0
tspan = (t0, 2.1)
ts = tspan[1]:0.01:tspan[2]
adtype = Optimization.AutoForwardDiff(;chunksize=32)
sampling_freq = 10.0

numStates = 4
ann_x_c0 = Float64[]
ann_x_d0 = copy(x_c0)
ann_x0 = [ann_x_c0..., ann_x_d0...]

hann_x_c0 = Float64[]
hann_x_d0 = copy(x_c0)
hann_x0 = [hann_x_c0..., hann_x_d0...]

# ANN model 
function setup_ann()
    
    _ann = Chain(Dense(numStates => 16, tanh),
                Dense(16 => numStates, identity)) 
    ann = HUDADE.LuxDiscreteModel(_ann, ann_x0, sampling_freq) # x_c0, because the state is automatically extended by the next time event point entry
    ann_p0 = ann.p

    sol_ann_before = solve(ann, Tsit5(); tspan=tspan)
    display(plot(sol_ann_before))

    reset!(100)
    
    function _myloss(p)
        loss(ann, p)
    end

    optf = Optimization.OptimizationFunction((x, p) -> _myloss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ann_p0)

    return ann, optprob, _myloss
end

# NODE
function setup_node()
    _node = Chain(Dense(4 => 16, tanh),
                Dense(16 => 2, identity)) 
    node = HUDADE.LuxSecondOrderNeuralODE(_node, x_c0)
    #node_p0 = node.p
    sol_node_before = solve(node; tspan=tspan)
    display(plot(sol_node_before))

    reset!(10)
    myloss(p) = loss(node, p)
    optf = Optimization.OptimizationFunction((x, p) -> myloss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, node.p)

    return node, optprob, myloss
end

# NODE (hybrid) model
function setup_hnode()
    _hnode = Chain(Dense(4 => 16, tanh),
                Dense(16 => 2, identity)) 
    hnode = HUDADE.LuxSecondOrderNeuralODE(_hnode, x_c0)
    hnode_p0 = hnode.p
    
    # rebuild!
    fct = HUDADE.rebuild(hnode.fct, c_x=c_x, a_x=a_x, z_len=4)
    hnode = HUDADE.HUDAODEProblem(fct, x_c0, x_d0, tspan)

    sol_hnode_before = solve(hnode; x0=x_c0, tspan=tspan, p=hnode_p0)
    display(plot(collect(u[1] for u in sol_hnode_before.u), collect(u[3] for u in sol_hnode_before.u)))

    reset!(10)

    function myloss(p)
        loss(hnode, p; x0=x_c0)
    end

    optf = Optimization.OptimizationFunction((x, p) -> myloss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, hnode_p0)
    
    return hnode, optprob, myloss
end

# ANN (hybrid) model
function setup_hann()
    _hann = Chain(Dense(4 => 16, tanh),
                Dense(16 => 4, identity)) 
    hann = HUDADE.LuxDiscreteModel(_hann, x_c0, sampling_freq) # x_c0, because the state is automatically extended by the next time event point entry
    hann_p0 = hann.p
    c_x_d = function(z, x_c, x_d, u, p)
        sx, vx, sy, vy, _ = x_d
        r = 0.1
        d = 0.9

        z[1] = 1.0+sx-r
        z[2] = 1.0-sx-r
        z[3] = 1.0+sy-r
        z[4] = 1.0-sy-r
        nothing
    end

    a_x_d = function(x_c_right, x_d_right, x_c_left, x_d_left, u, p, t, idx)
        sx_left, vx_left, sy_left, vy_left, _ = x_d_left 
        eps = 1e-16
        r = 0.1
        d = 0.9

        # default setup
        x_c_right[:] = x_c_left
        x_d_right[1] = sx_left 
        x_d_right[2] = vx_left
        x_d_right[3] = sy_left 
        x_d_right[4] = vy_left
        x_d_right[5] = x_d_left[5]
    
        if idx == 1 # state event (left)
            x_d_right[1] = -1.0+(r+eps)
            x_d_right[2] = -vx_left*d
        elseif idx == 2 # state event (right)
            x_d_right[1] = 1.0-(r+eps) 
            x_d_right[2] = -vx_left*d
        elseif idx == 3 # state event (bottom)
            x_d_right[3] = -1.0+(r+eps)
            x_d_right[4] = -vy_left*d
        elseif idx == 4 # state event (top)
            x_d_right[3] = 1.0-(r+eps)
            x_d_right[4] = -vy_left*d
        end

        nothing
    end

    orig_a_t = hann.fct.a_t
    hann_a_t = function(x_c_right, x_d_right, x_c_left, x_d_left, u, p, t)
        
        z_len = 4
        
        # check before
        z_before = similar(x_c_left, z_len)
        c_x_d(z_before, x_c_left, x_d_left, u, p)

        # evaluate a_t
        orig_a_t(x_c_right, x_d_right, x_c_left, x_d_left, u, p, t)
        
        # check after
        z_after = similar(x_c_left, z_len)
        c_x_d(z_after, x_c_right, x_d_right, u, p)

        # check for events
        for idx in 1:z_len
            if sign(z_after[idx]) != sign(z_before[idx])
                a_x_d(x_c_right, x_d_right, x_c_right, x_d_right, u, p, t, idx)
            end
        end
        
        nothing
    end

    hann.fct = HUDADE.rebuild(hann.fct; a_t=hann_a_t)
    # [ToDo] Time event ausführen und im TimeEvent checken, ob c_x (und falls ja) dann a_x ausführen
    sol_hann_before = solve(hann; tspan=tspan, p=hann_p0) # , x0=[hann_x_c0..., hann_x_d0...])
    display(plot(sol_hann_before))
    reset!(100)
    
    function myloss(p)
        loss(hann, p) # ; x0=[hann_x_c0..., hann_x_d0...])
    end

    optf = Optimization.OptimizationFunction((x, p) -> myloss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, hann_p0)

    return hann, optprob, myloss
end

global horizon, iteration

# functions
global u_latest = nothing
global u_pre_latest
function callback(state, val)
    global horizon, iteration, u_pre_latest, u_latest

    if VERSION >= v"1.10"
        state = state.u
    end

    if val < 0.06 && horizon < length(ts)
        horizon += 1
    end

    if iteration%10 == 0
        @info "Iteration: $(iteration) | Value: $(round(val; digits=6)) | Horizon: $(round(horizon/length(ts)*100; digits=1))%"
    end

    iteration += 1
    u_pre_latest = u_latest
    u_latest = copy(state)

    return false
end

function loss(prob::HUDAODEProblem, p; x0=nothing)
    tspan = (ts[1], ts[horizon])
    saveat = ts[1:horizon]
    sol = solve(prob, Tsit5(); x0=x0, tspan=tspan, p=p, saveat=saveat)

    return sum(abs.(collect(u[1] for u in sol.u) .- data1[1:horizon]))/horizon + 
        sum(abs.(collect(u[3] for u in sol.u) .- data3[1:horizon]))/horizon
end

function reset!(hor::Int)
    global horizon, iteration 
    iteration = 0

    horizon = round(Int, round(Int, length(ts)/100*hor))
end

function train(optprob, eta, maxiters)
    global u_pre_latest
    result = Optimization.solve(optprob, OptimizationOptimisers.Adam(eta); callback=callback, maxiters=maxiters)
    optprob = remake(optprob; u0=u_pre_latest)
    return optprob
end

c_x = function(z, x_c, x_d, u, p)
    sx, vx, sy, vy = x_c 
    r = 0.1
    d = 0.9

    z[1] = 1.0+sx-r
    z[2] = 1.0-sx-r
    z[3] = 1.0+sy-r
    z[4] = 1.0-sy-r
    nothing
end

a_x = function(x_c_right, x_d_right, x_c_left, x_d_left, u, p, t, idx)
    sx_left, vx_left, sy_left, vy_left = x_c_left 
    eps = 1e-16
    r = 0.1
    d = 0.9

    # default setup
    x_c_right[1] = sx_left 
    x_c_right[2] = vx_left
    x_c_right[3] = sy_left 
    x_c_right[4] = vy_left
   
    if idx == 1 # state event (left)
        x_c_right[1] = -1.0+(r+eps)
        x_c_right[2] = -vx_left*d
    elseif idx == 2 # state event (right)
        x_c_right[1] = 1.0-(r+eps) 
        x_c_right[2] = -vx_left*d
    elseif idx == 3 # state event (bottom)
        x_c_right[3] = -1.0+(r+eps)
        x_c_right[4] = -vy_left*d
    elseif idx == 4 # state event (top)
        x_c_right[3] = 1.0-(r+eps)
        x_c_right[4] = -vy_left*d
    end

    nothing
end

using HUDADE: BouncingBall2D

# Ground Truth 
gt = BouncingBall2D(; x_c0=x_c0, x_d0=x_d0, tspan=tspan)
solution_gt = solve(gt; saveat=ts)
plot(ts, collect(solution_gt(t)[1] for t in ts))
plot!(ts, collect(solution_gt(t)[2] for t in ts))
data1 = collect(solution_gt(t)[1] for t in ts)
data3 = collect(solution_gt(t)[3] for t in ts)
#data1_32 = Float32.(data1)
#data3_32 = Float32.(data3)
plot(data1, data3)
 
# ANN model
ann, optprob, myloss = setup_ann()

sol = solve(ann; tspan=tspan)
plot(sol)
plot(sol.t, collect(u[5] for u in sol.u))

myloss(optprob.u0)
optprob = train(optprob, 1e-2, 2000) 
myloss(optprob.u0)
optprob = train(optprob, 1e-3, 1000)
myloss(optprob.u0)
optprob = train(optprob, 1e-4, 1000)
myloss(optprob.u0)

sol_ann_after = solve(ann; tspan=tspan, p=optprob.u0, saveat=ts)
plot(collect(u[1] for u in sol_ann_after.u), collect(u[3] for u in sol_ann_after.u))
plot!(data1, data3)

# ANN (hybrid) model
hann, optprob, myloss = setup_hann()

myloss(optprob.u0)
optprob = train(optprob, 1e-2, 1000)
myloss(optprob.u0)
optprob = train(optprob, 1e-3, 2000)
myloss(optprob.u0)
optprob = train(optprob, 1e-4, 2000)
myloss(optprob.u0)

sol_hann_after = solve(hann; tspan=tspan, p=optprob.u0, saveat=ts)
plot(collect(u[1] for u in sol_hann_after.u), collect(u[3] for u in sol_hann_after.u))
plot!(data1, data3)

# NODE model
node, optprob, myloss = setup_node()

myloss(optprob.u0)
optprob = train(optprob, 1e-2, 2500) # 2500
myloss(optprob.u0)
optprob = train(optprob, 1e-3, 2500) # 2500
myloss(optprob.u0)

sol_node_after = solve(node; tspan=tspan, p=optprob.u0, saveat=ts)
plot(collect(u[1] for u in sol_node_after.u), collect(u[3] for u in sol_node_after.u))
plot!(data1, data3)

# NODE (hybrid) model
hnode, optprob, myloss = setup_hnode()

myloss(optprob.u0)
optprob = train(optprob, 1e-2, 500)
myloss(optprob.u0)

#s = solve(hnode; tspan=tspan, p=optprob.u0)
#plot(s)

sol_hnode_after = solve(hnode,; tspan=tspan, p=optprob.u0, saveat=ts)
plot(collect(u[1] for u in sol_hnode_after.u), collect(u[3] for u in sol_hnode_after.u))
plot!(data1, data3)

# plot all results 
colors = Colors.distinguishable_colors(4, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

fig = plotBB(; legend=:topleft)
plot!(fig, data1, data3; dpi=300, label="ground truth", color=:black)
plot!(fig, collect(u[1] for u in sol_ann_after.u), collect(u[3] for u in sol_ann_after.u); label="ANN", color=colors[1])
plot!(fig, collect(u[1] for u in sol_hann_after.u), collect(u[3] for u in sol_hann_after.u); label="hy. ANN", color=colors[2])
plot!(fig, collect(u[1] for u in sol_node_after.u), collect(u[3] for u in sol_node_after.u); label="neu. ODE", color=colors[3])
plot!(fig, collect(u[1] for u in sol_hnode_after.u), collect(u[3] for u in sol_hnode_after.u); label="hy. neu. ODE", color=colors[4])

scatterBB!(fig)

savefig(fig, joinpath(@__DIR__, "plots", "pdf", "common.pdf"))
display(fig)