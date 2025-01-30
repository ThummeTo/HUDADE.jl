#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module FMIBaseExt

using HUDADE, FMIBase

function HUDADE.FMUModel(fmu::FMU)

    fct = HUDAODEFunction()

    # functions 
    fct.α = function(x_c, x_d, u, p, t)
        comp = nothing
        comp, x0_c = FMIBase.prepareSolveFMU(fmu, comp)
        x_c[:] = x0_c
        nothing
    end

    fct.ω = function()
        comp = nothing # [ToDo] this must be the current component to allow for cleanup!
        finishSolveFMU(fmu, comp)
        nothing
    end

    fct.f = function(ẋ_c, x_c, x_d, u, p, t)
        fmu(; dx=ẋ_c, 
            dx_refs=:derivatives, 
            x=x_c, 
            #u=u, # [ToDo: inputs!]
            #u_refs=u_refs,
            p=p, 
            p_refs=:parameters, 
            t=t)
        nothing
    end

    fct.g = function(y, x_c, x_d, u, p, t)
        fmu(; y=y, 
            y_refs=:outputs, 
            x=x_c, 
            #u=u, # [ToDo: inputs!]
            #u_refs=u_refs,
            p=p, 
            p_refs=:parameters, 
            t=t)
        nothing
    end

    fct.c_x = function(z, x_c, x_d, u, p)
        #c = nothing # [todo]
        #condition!(c::FMUInstance, ec, x::AbstractArray{<:Real}, t::Real, inputFunction::Union{Nothing,FMUInputFunction},)

        fmu(; x=x_c, 
            #u=u, # [ToDo: inputs!]
            #u_refs=u_refs,
            p=p, 
            p_refs=:parameters, 
            t=t,
            ec=z,
            ec_ids=:all)
        nothing
    end

    fct.a_x = function(x_c_right_local, x_d_right_local, x_c_left, x_d_left, u, p, t, idx)
        # [ToDo] for idx == -1 this returns just the current state, for idx >= 1 the current state change by event
        nothing
    end

    fct.c_t = function(t_next, x_d, u, p, t)
        #c = nothing # [todo]
        #time_choice(c::FMU2Component, integrator, tStart, tStop)
        nothing
    end

    fct.a_t = function(x_c_right, x_d_right, x_c_left, x_d_left, u, p, t)
        # [ToDo] 
        nothing
    end
    
    return fct
end

end # FMIBaseExt