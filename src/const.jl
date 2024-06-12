#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

function c_t_NO_FUNCTION(t_next, x_d, u, p, t)
    nothing
end

function c_x_NO_FUNCTION(z, x_c, x_d, u, p)
    nothing
end

function a_t_NO_FUNCTION(x_c_right, x_d_right, x_c_left, x_d_left, u, p, t)
    nothing 
end

function a_x_NO_FUNCTION(x_c_right, x_d_right, x_c_left, x_d_left, u, p, t, idx)
    nothing 
end

function f_NO_FUNCTION(ẋ_c, x_c, x_d, u, p, t)
    nothing
end

function g_NO_FUNCTION(y, x_c, x_d, u, p, t)
    nothing
end

function α_NO_FUNCTION(x_c, x_d, u, p, t)
    nothing
end

function ω_NO_FUNCTION()
    nothing
end

function ϵ_NO_FUNCTION(ϵ, x_c, x_d, u, p, t)
    nothing
end

NoVector = Vector{Float64}()