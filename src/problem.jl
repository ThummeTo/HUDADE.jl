#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import SciMLSensitivity.SciMLBase
import SciMLSensitivity.SciMLBase: ODEProblem, RightRootFind
import DiffEqCallbacks: IterativeCallback, VectorContinuousCallback, CallbackSet
using Optim

# [todo] remove!
using SciMLSensitivity.ForwardDiff
using SciMLSensitivity.Zygote: ignore_derivatives

mutable struct HUDAODEProblem 
    fct::HUDAODEFunction
    problem::ODEProblem 
    callbacks::Vector{Any} # [ToDo] correct vector type!
    currentSolution::HUDADESolution
    
    function HUDAODEProblem(fct::HUDAODEFunction, x0_c, x0_d, tspan; p=Float64[], interp_points::Int=10, tType=Float64)

        inst = new()
        inst.fct = fct

        #fct.input.x[1][:] = x0_c
        #fct.input.x[2][:] = x0_d

        # we wrap the DiffEq `f` into a custom `f`, supporting discrete state, parameters and inputs
        function f(ẋ, x, p, t)

            x_c, x_d, u_c, u_d = fct.x_unpack(x)
            
            ẋ_c, _, _, _ = fct.x_unpack(ẋ)
           
            # [todo] extend to discrete inputs, add continuous input function
            u = u_c

            fct.f(ẋ_c, x_c, x_d, u, p, t)
            
            # todo: make above as view!
            ẋ[1:length(ẋ_c)] = ẋ_c

            nothing
        end

        # todo: add inputs!
        x0 = fct.x_pack(x0_c, x0_d, Float64[], Float64[])

        problem = ODEProblem(f, x0, tspan, p)

        callbacks = []

        if fct.c_t != c_t_NO_FUNCTION && fct.a_t != a_t_NO_FUNCTION

            function a_t(integrator)

                # ToDo: Is this correct? Pulling from the integrator?
                t = integrator.t
                x_c_left, x_d_left, u_c_left, u_d_left = fct.x_unpack(integrator.u)  
                
                # ToDo: Remove copy here!
                x_c_right_local = copy(x_c_left)
                x_d_right_local = copy(x_d_left)
    
                # todo: correct inputs 
                u = u_c_left
    
                fct.a_t(x_c_right_local, x_d_right_local, x_c_left, x_d_left, u, p, t)
    
                needoptim = true
                if needoptim 
                    # find global x_c_right for local x_c_right

                    # allocate buffer
                    x_c_local_buffer = copy(x_c_right_local)
                    x_d_local_buffer = copy(x_d_right_local)

                    function res(_x_right)
                        _x_c_right, _x_d_right = fct.x_unpack(_x_right) # [ToDo] unpack is only valid w.o. inputs!  
                        fct.a_t(x_c_local_buffer, x_d_local_buffer, _x_c_right, _x_d_right, u, p, t) # -1 = ignore event handling
                        return sum(abs.(x_c_local_buffer - x_c_right_local)) + sum(abs.(x_d_local_buffer - x_d_right_local))
                    end

                    result = Optim.optimize(res, [x_c_right_local..., x_d_right_local...], Optim.BFGS())

                    if result.minimum > 1e-6
                        @error "Found no state at t=$(t), residual is $(result.minimum)"
                    end

                    x_c_right, x_d_right = fct.x_unpack(result.minimizer)
        
                    integrator.u[:] = fct.x_pack(x_c_right, x_d_right, u, u)
                end

                add_event!(inst.currentSolution, t, 0)
    
                nothing
            end

            function c_t(integrator)
                x_c, x_d, u_c, u_d = fct.x_unpack(integrator.u)
                t = integrator.t

                u = u_c # todo!

                t_next = [t] # todo!
                fct.c_t(t_next, x_d, u, p, t)
                return t_next[1]
            end

            timeEventCb = IterativeCallback(c_t,
                a_t, 
                tType; 
                initial_affect=false, 
                save_positions=(false, false))
            push!(callbacks, timeEventCb)
        end

        if fct.c_x != c_x_NO_FUNCTION && fct.a_x != a_x_NO_FUNCTION

            function a_x(integrator, idx)

                #@info "$(typeof(integrator.t))"

                # ToDo: Is this correct? Pulling from the integrator?
                t = integrator.t
                x_c_left, x_d_left, u_c_left, u_d_left = fct.x_unpack(integrator.u)  
                
                # ToDo: Remove copy here!
                x_c_right_local = copy(x_c_left)
                x_d_right_local = copy(x_d_left)
    
                # todo: correct inputs 
                u = u_c_left
    
                fct.a_x(x_c_right_local, x_d_right_local, x_c_left, x_d_left, u, p, t, idx)

                needoptim = true
                if needoptim 
                    # find global x_c_right for local x_c_right

                    # allocate buffer
                    x_c_local_buffer = copy(x_c_right_local)
                    x_d_local_buffer = copy(x_d_right_local)

                    function res(_x_right)
                        _x_c_right, _x_d_right = fct.x_unpack(_x_right) # [ToDo] unpack is only valid w.o. inputs!  
                        fct.a_x(x_c_local_buffer, x_d_local_buffer, _x_c_right, _x_d_right, u, p, t, -1) # -1 = ignore event handling
                        return sum(abs.(x_c_local_buffer - x_c_right_local)) + sum(abs.(x_d_local_buffer - x_d_right_local))
                    end

                    result = Optim.optimize(res, [x_c_right_local..., x_d_right_local...], Optim.BFGS())

                    if result.minimum > 1e-6
                        @error "Found no state at t=$(t), residual is $(result.minimum)"
                    end

                    x_c_right, x_d_right = fct.x_unpack(result.minimizer)
        
                    integrator.u[:] = fct.x_pack(x_c_right, x_d_right, u, u)
                end

                add_event!(inst.currentSolution, t, idx)
    
                #fct.input.x[1][:] = x_c_right
                #fct.input.x[2][:] = x_d_right
    
                #fct.input_p[:] = fct.p_pack(u, p)
                #@info "Affect: $(x_d_right)"
                #fct.input = ArrayPartition(x_c_right, x_d_right, u, p, t)
    
                #if isnothing(fct.parent) # function is root!
                #    integrator.u[:] = fct.x_pack(x_c_right, x_d_right, u, u) # todo: correct inputs!
                #else
                    # [ToDo] Trigger a recursion for optimization of the right new global state
                #    @warn "Propagation for new right state currently not implemented!"
                #    integrator.u[:] = fct.x_pack(x_c_right, x_d_right, u, u) # todo: correct inputs!
                #end
    
                nothing
            end

            function c_x(z, x, t, integrator)
                x_c, x_d, u_c, u_d = fct.x_unpack(x)

                # if z doesnt allow for sensitivities
                # if !isa(z, AbstractArray{<:ForwardDiff.Dual})
                #     x_c = ForwardDiff.value.(x_c)
                #     p = ForwardDiff.value.(p)
                # end

                u = u_c # todo!

                fct.c_x(z, x_c, x_d, u, p)
                nothing
            end

            stateEventCb = VectorContinuousCallback(c_x,
                a_x, 
                fct.z_len;
                rootfind=RightRootFind,
                save_positions=(false, false),
                interp_points=interp_points)
            push!(callbacks, stateEventCb)
        end

        inst.problem = problem 
        inst.callbacks = callbacks

        return inst
    end
end
export HUDAODEProblem 

function initialize(prob::HUDAODEProblem, tspan)

    x_c = nothing
    x_d = nothing

    ignore_derivatives() do
        x_c, x_d, u_c, u_d = prob.fct.x_unpack(prob.problem.u0)
        p = prob.problem.p
        t = tspan[1]

        u = u_c # todo!

        prob.fct.α(x_c, x_d, u, p, t)
    end

    return x_c, x_d
end

function cleanup(prob::HUDAODEProblem)
    ignore_derivatives() do
        prob.fct.ω()
    end

    return nothing
end

function SciMLBase.solve(prob::HUDAODEProblem, args...; tspan=prob.problem.tspan, p=prob.problem.p, callbacks::Bool=true, kwargs...)

    x_c0, x_d0 = initialize(prob, tspan)
    
    x0 = prob.fct.x_pack(x_c0, x_d0, Float64[], Float64[]) # todo: inputs!

    if length(x0) == 0
        x0 = [0.0]  
    end

    prob.currentSolution = HUDADESolution()

    callback = nothing 
    if callbacks 
        callback = CallbackSet(prob.callbacks...)
    end

    prob.currentSolution.solution = SciMLBase.solve(prob.problem, args...; tspan=tspan, p=p, u0=x0, callback=callback, kwargs...)

    cleanup(prob)
    
    return prob.currentSolution
end