#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# this file containes placeholders for sensitivity (AD) related functions

# check if scalar/vector is ForwardDiff.Dual
function isdual(e)
    return false 
end
function isdual(e::Tuple)
    return any(isdual.(e))
end

# check if scalar/vector is ForwardDiff.Dual
function istracked(e)
    return false 
end
function istracked(e::Tuple)
    return any(istracked.(e))
end

# makes Reals from ForwardDiff.Dual scalar/vector
function undual(e::AbstractArray)
    return undual.(e)
end
function undual(e::AbstractArray{Float64})
    return e
end
function undual(e::Tuple)
    if !isdual(e)
        return e 
    end
    return undual.(e)
end
function undual(::Nothing)
    return nothing
end
function undual(e)
    return e
end

# makes Reals from ReverseDiff.TrackedXXX scalar/vector
function untrack(e::AbstractArray)
    return untrack.(e)
end
function untrack(e::AbstractArray{Float64})
    return e
end
function untrack(e::Tuple)
    if !istracked(e)
        return e 
    end
    return untrack.(e)
end
function untrack(::Nothing)
    return nothing
end
function untrack(e)
    return e
end

# makes Reals from ForwardDiff/ReverseDiff.TrackedXXX scalar/vector
function unsense(e::AbstractArray)
    return unsense.(e)
end
function unsense(e::AbstractArray{Float64})
    return e
end
function unsense(e::Tuple)
    if !isdual(e) && !istracked(e)
        return e 
    end
    return unsense.(e)
end
function unsense(::Nothing)
    return nothing
end
function unsense(e)
    return e
end

# makes copied Reals from ForwardDiff/ReverseDiff.TrackedXXX scalar/vector
function unsense_copy(e) 
    return unsense(e) # inherit, most `unsense` dispatches are allocating anyway
end
function unsense_copy(e::AbstractArray{Float64})
    return copy(e)
end

# set sensitive primitives (this is intentionally NO additional dispatch for `setindex!`) 
function sense_setindex!(A, x, i)
    return setindex!(A, x, i)
end