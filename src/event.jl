#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

struct HUDADEEvent{T}
    t::T
    idx::UInt32
    
    function HUDADEEvent{T}(t::T, idx::UInt32) where {T}
        return new(t, idx)
    end
end
export HUDADEEvent