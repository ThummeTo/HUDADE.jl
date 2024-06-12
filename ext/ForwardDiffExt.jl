#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module ForwardDiffExt

using HUDADE, ForwardDiff

# check if scalar/vector is ForwardDiff.Dual
function HUDADE.isdual(e::ForwardDiff.Dual{T, V, N}) where {T, V, N}
    return true
end
function HUDADE.isdual(e::AbstractVector{<:ForwardDiff.Dual{T, V, N}}) where {T, V, N}
    return true
end

# makes Reals from ForwardDiff.Dual scalar/vector
function HUDADE.undual(e::ForwardDiff.Dual)
    return ForwardDiff.value(e)
end

# makes Reals from ForwardDiff scalar/vector
function HUDADE.unsense(e::ForwardDiff.Dual)
    return ForwardDiff.value(e)
end

# set sensitive primitives (this is intentionally NO additional dispatch for `setindex!`) 
function HUDADE.sense_setindex!(A::Vector{Float64}, x::ForwardDiff.Dual, i::Int64)
    return setindex!(A, undual(x), i)
end

# specials 
HUDADE.add_event!(solution::HUDADESolution, t::ForwardDiff.Dual, idx) = HUDADE.add_event!(solution, ForwardDiff.value(t), idx)

end # ForwardDiffExt

