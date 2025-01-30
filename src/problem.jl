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
    p_solve # ToDo: Maybe this is the same as problem.p ?
    
    function HUDAODEProblem(fct::HUDAODEFunction, x0_c::AbstractVector{T}, x0_d, tspan::Tuple{T, T}; p=T[], interp_points::Int=10) where {T}

        inst = new()
        inst.fct = fct
        inst.p_solve = p # initial value

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
            len_dx = length(ẋ_c)
            ẋ[1:len_dx] = ẋ_c
            ẋ[len_dx+1:end] .= T(0.0)

            nothing
        end

        # todo: add inputs!
        x0 = fct.x_pack(x0_c, x0_d, similar(x0_c, 0), similar(x0_d,0))

        problem = ODEProblem(f, x0, tspan, p)

        callbacks = []

        if fct.c_t != c_t_NO_FUNCTION && fct.a_t != a_t_NO_FUNCTION

            function a_t(integrator)

                p = inst.p_solve

                # ToDo: Is this correct? Pulling from the integrator?
                t_left = integrator.t
                t_right = t_left # is this correct?
                x_c_left, x_d_left, u_c_left, u_d_left = fct.x_unpack(integrator.u)  
                
                # ToDo: Remove copy here!
                x_c_right_local = deepcopy(x_c_left)
                x_d_right_local = deepcopy(x_d_left)
    
                # todo: correct inputs 
                u = u_c_left
    
                fct.a_t(x_c_right_local, x_d_right_local, x_c_left, x_d_left, u, p, t_left, true)
    
                #@info "true -> $((x_c_right_local, x_d_right_local))"

                needoptim = true
                if needoptim 
                    # find global x_c_right for local x_c_right

                    # allocate buffer
                    x_c_local_buffer = deepcopy(x_c_right_local)
                    x_d_local_buffer = deepcopy(x_d_right_local)

                    function res(_x_right)
                        _x_c_right, _x_d_right = fct.x_unpack(_x_right) # [ToDo] unpack is only valid w.o. inputs!  
                        
                        x_c_local_buffer[:] = _x_c_right
                        x_d_local_buffer[:] = _x_d_right

                        fct.a_t(x_c_local_buffer, x_d_local_buffer, _x_c_right, _x_d_right, u, p, t_right, false) # false = ignore event handling
                        return sum(abs.(x_c_local_buffer - x_c_right_local)) + sum(abs.(x_d_local_buffer - x_d_right_local))
                    end

                    #@info "Before: $(typeof([x_c_left..., x_d_left...]))"

                    # todo: why not [x_c_left..., x_d_left...]
                    result = Optim.optimize(res, [x_c_right_local..., x_d_right_local...]; method=Optim.BFGS()) #, allow_f_increases=true)

                    if result.minimum > 1e-6 # ToDo: This is too big.
                        @error "Found no state at t=$(t_right).\nFor local state: $([x_c_right_local..., x_d_right_local...])\nResidual is $(result.minimum).\nBest state is: $(result.minimizer)."
                    end

                    x_c_right, x_d_right = fct.x_unpack(result.minimizer)
        
                    integrator.u[:] = fct.x_pack(x_c_right, x_d_right, u, u)

                    #@info "After: $(typeof(x_c_right))"
                end

                #@info "$((x_c_right, x_d_right)) $(result.minimum)"

                add_event!(inst.currentSolution, t_right, 0)
    
                nothing
            end

            function c_t(integrator)

                p = inst.p_solve

                x_c, x_d, u_c, u_d = fct.x_unpack(integrator.u)
                t = integrator.t

                u = u_c # todo!

                #t_next = [similar(x_c)] # todo!
                #fct.c_t(t_next, x_d, u, p, t)
                #return t_next[1]

                # [Todo] check if type inconsistency and only FD.value if necessary!
                t_next = fct.c_t(x_d, u, p, t)
                @assert !isdual(t_next) "The function c_t returned a FD.Dual, which is not supported.\nThis may be because you defined your time event depending on the state."
                return t_next # undual(t_next)
            end

            timeEventCb = IterativeCallback(c_t,
                a_t, 
                T; 
                initial_affect=false, 
                save_positions=(false, false))
            push!(callbacks, timeEventCb)
        end

        if fct.c_x != c_x_NO_FUNCTION && fct.a_x != a_x_NO_FUNCTION

            function a_x(integrator, idx)

                p = inst.p_solve

                #@info "$(typeof(integrator.t))"

                # ToDo: Is this correct? Pulling from the integrator?
                t = integrator.t
                x_c_left, x_d_left, u_c_left, u_d_left = fct.x_unpack(integrator.u)  
                
                # ToDo: Remove copy here!
                x_c_right_local = deepcopy(x_c_left)
                x_d_right_local = deepcopy(x_d_left)
    
                # todo: correct inputs 
                u = u_c_left
    
                fct.a_x(x_c_right_local, x_d_right_local, x_c_left, x_d_left, u, p, t, idx)

                needoptim = true
                if needoptim 
                    # find global x_c_right for local x_c_right

                    # allocate buffer
                    x_c_local_buffer = deepcopy(x_c_right_local)
                    x_d_local_buffer = deepcopy(x_d_right_local)
                    
                    function res(_x_right)
                        _x_c_right, _x_d_right = fct.x_unpack(_x_right) # [ToDo] unpack is only valid w.o. inputs!  
                        fct.a_x(x_c_local_buffer, x_d_local_buffer, _x_c_right, _x_d_right, u, p, t, -1) # -1 = ignore event handling
                        return sum(abs.(x_c_local_buffer - x_c_right_local)) + sum(abs.(x_d_local_buffer - x_d_right_local))
                    end

                    # why not: [x_c_left..., x_d_left...]
                    result = Optim.optimize(res, [x_c_right_local..., x_d_right_local...], Optim.BFGS())

                    if result.minimum > 1e-6
                        @error "Found no state at t=$(t).\nFor local state: $([x_c_right_local..., x_d_right_local...])\nResidual is $(result.minimum).\nBest state is: $(result.minimizer)."
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

                p = inst.p_solve
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

function Base.hasproperty(obj::HUDAODEProblem, var::Symbol)
    if var ∈ (:p,)
        return true 
    else
        return Base.hasfield(obj, var)
    end
end

function Base.getproperty(obj::HUDAODEProblem, var::Symbol)
    if var ∈ (:p,)
        return Base.getfield(obj.problem, var) 
    else
        return Base.getfield(obj, var)
    end
end

function Base.setproperty!(obj::HUDAODEProblem, var::Symbol, val)
    if var ∈ (:p,)
        return Base.setfield!(obj.problem, var, val) 
    else
        return Base.setfield!(obj, var, val)
    end
end

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

function SciMLBase.solve(prob::HUDAODEProblem, args...; tspan=prob.problem.tspan, p=prob.problem.p, callbacks::Bool=true, x0=nothing, u0=nothing, kwargs...)

    @assert isnothing(u0) "u0 is not defined for HUDADEs (could be easily mixed up with the input u), please use x0 for the inital state instead."

    if isnothing(x0)
            x_c0, x_d0 = initialize(prob, tspan)
        
        x0 = prob.fct.x_pack(x_c0, x_d0, similar(x_c0,0), similar(x_d0,0)) # todo: inputs!

        if length(x0) == 0
            x0 = [0.0f0] # zero state system 
        end
    end

    prob.p_solve = p # to get `p` into the callbacks!
    prob.currentSolution = HUDADESolution()

    callback = nothing 
    if callbacks 
        callback = CallbackSet(prob.callbacks...)
    end

    prob.currentSolution.solution = SciMLBase.solve(prob.problem, args...; tspan=tspan, p=p, u0=x0, callback=callback, kwargs...)

    cleanup(prob)
    
    return prob.currentSolution
end
