#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using SciMLSensitivity.ForwardDiff

# check if scalar/vector is ForwardDiff.Dual
function isdual(e::ForwardDiff.Dual{T, V, N}) where {T, V, N}
    return true
end
function isdual(e::AbstractVector{<:ForwardDiff.Dual{T, V, N}}) where {T, V, N}
    return true
end

# makes Reals from ForwardDiff.Dual scalar/vector
function undual(e::ForwardDiff.Dual)
    return ForwardDiff.value(e)
end

# makes Reals from ForwardDiff scalar/vector
function unsense(e::ForwardDiff.Dual)
    return ForwardDiff.value(e)
end

# set sensitive primitives (this is intentionally NO additional dispatch for `setindex!`) 
function sense_setindex!(A::Vector{Float64}, x::ForwardDiff.Dual, i::Int64)
    return setindex!(A, undual(x), i)
end

# specials 
add_event!(solution::HUDADESolution, t::ForwardDiff.Dual, idx) = add_event!(solution, ForwardDiff.value(t), idx)
