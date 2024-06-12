#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module FluxExt

using HUDADE, Flux

function HUDADE.FluxModel(model)

    fct = HUDAODEFunction()

    _p, re = Flux.destructure(model)

    fct.parameters = _p

    # functions 
    fct.g = function(y, x_c, x_d, u, p, t)
        y[:] = re(p)(u)
        nothing
    end
    
    return fct
end

end # FluxExt