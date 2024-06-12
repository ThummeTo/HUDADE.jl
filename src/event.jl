#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

struct HUDADEEvent
    t::Float64
    idx::UInt32
    
    function HUDADEEvent(t::Float64, idx::UInt32)
        return new(t, idx)
    end
end
export HUDADEEvent