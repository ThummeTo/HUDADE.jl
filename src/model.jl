#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import SciMLSensitivity.SciMLBase
using  SciMLSensitivity.SciMLBase.RecursiveArrayTools

abstract type AbstractHUDAODEModel end

struct HUDAODEModel <: AbstractHUDAODEModel
 
    # functions
    fct::HUDAODEModel
    p

    function HUDAODEModel(fct::HUDAODEFunction, p)
        
        return new(fct, p)
    end
end
export HUDAODEModel

function SciMLBase.solve(model::HUDAODEModel, x0_c, x0_d, tspan; tType=Float64)

    problem = HUDAODEProblem(model.fct, x0_c, x0_d, tspan; p=model.p, tType=tType)
    return SciMLBase.solve(problem, args...; p=p, kwargs...)
end