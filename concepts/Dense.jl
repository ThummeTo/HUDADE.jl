#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

function Dense(W, b, σ)

    fct = HUDAODEFunction(; name="Dense")

    size_W = size(W)
    len_W = size_W[1]*size_W[2]

    function p_pack(_W, _b)
        vcat(vec(_W), _b) 
    end
    function p_unpack(container)
        return (reshape(container[1:len_W], sizeW),
                container[len_W+1:end])
    end

    fct.parameters = p_pack(W, b)

    # functions 
    fct.g = function(y, x_c, x_d, u, p, t)
        W, b = p_unpack(p)
        y[:] = σ.(W*u .+ b)
        nothing
    end
    
    return fct
end