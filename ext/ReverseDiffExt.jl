#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module ReverseDiffExt

using HUDADE, ReverseDiff

# check if scalar/vector is ReverseDiff.TrackedReal
function HUDADE.istracked(e::ReverseDiff.TrackedReal) 
    return true
end
function HUDADE.istracked(e::AbstractVector{<:ReverseDiff.TrackedReal}) 
    return true
end
function HUDADE.istracked(e::ReverseDiff.TrackedArray) 
    return true
end

# makes Reals from ReverseDiff.TrackedXXX scalar/vector
function HUDADE.untrack(e::ReverseDiff.TrackedReal)
    return ReverseDiff.value(e)
end
function HUDADE.untrack(e::ReverseDiff.TrackedArray)
    return ReverseDiff.value(e)
end

# makes Reals from ReverseDiff.TrackedXXX scalar/vector
function HUDADE.unsense(e::ReverseDiff.TrackedReal)
    return ReverseDiff.value(e)
end
function HUDADE.unsense(e::ReverseDiff.TrackedArray)
    return ReverseDiff.value(e)
end

# set sensitive primitives (this is intentionally NO additional dispatch for `setindex!`) 
function HUDADE.sense_setindex!(A::Vector{Float64}, x::ReverseDiff.TrackedReal, i::Int64)
    return setindex!(A, untrack(x), i)
end

end # ReverseDiffExt

