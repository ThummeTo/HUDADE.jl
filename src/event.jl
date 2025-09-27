#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

struct HUDADEEvent{T}
    t::T
    idx::UInt32

    x_left::Vector{Float64}
    x_right::Vector{Float64}
    
    function HUDADEEvent{T}(t::T, idx::UInt32, x_left::Vector{Float64}, x_right::Vector{Float64}) where {T}
        return new(t, idx, x_left, x_right)
    end
end
export HUDADEEvent