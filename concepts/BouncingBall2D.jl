#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

#    [  m,    g,   r,   d]
p0 = [1.0, 9.81, 0.1, 0.9]
u0 = zeros(Float64, 0)
t0 = 0.0

z0 = zeros(4)

#      [s_x, v_x, s_y, v_y]
x_c0 = [0.0, 1.0, 0.0, 0.0]
#      [n]
x_d0 = [0.0]

function BouncingBall2D(; m=1.0, g=9.81, r=0.1, d=0.9, x_c0=[0.0, 1.0, 0.0, 0.0], x_d0=[0.0])
    p0 = [m, g, r, d]
    u0 = zeros(Float64, 0)
    t0 = 0.0

    α = function(x_c, x_d, u, p, t)
        x_c[1:4] = x_c0
        x_d[1] = x_d0[1]
        nothing
    end

    f = function(ẋ_c, x_c, x_d, u, p, t)
        sx, vx, sy, vy = x_c 
        m, g, r, d = p 
        n = x_d[1]

        ax = 0.0
        ay = -g 

        ẋ_c[1] = vx 
        ẋ_c[2] = ax
        ẋ_c[3] = vy
        ẋ_c[4] = ay
        nothing
    end

    c_x = function(z, x_c, x_d, u, p)
        sx, vx, sy, vy = x_c 
        m, g, r, d = p

        z[1] = 1.0+sx-r
        z[2] = 1.0-sx-r
        z[3] = 1.0+sy-r
        z[4] = 1.0-sy-r
        nothing
    end

    a_x = function(x_c_right, x_d_right, x_c_left, x_d_left, u, p, t, idx)
        sx_left, vx_left, sy_left, vy_left = x_c_left 
        m, g, r, d = p
        n_left = x_d_left[1]
        eps = 1e-16

        # default setup
        x_c_right[1] = sx_left 
        x_c_right[2] = vx_left
        x_c_right[3] = sy_left 
        x_c_right[4] = vy_left
        x_d_right[1] = n_left
        
        if idx == 0 # time event 
        
        elseif idx == 1 # state event (left)
            x_d_right[1] = n_left+1
            x_c_right[1] = -1.0+(r+eps)
            x_c_right[2] = -vx_left*d
        elseif idx == 2 # state event (right)
            x_d_right[1] = n_left+1
            x_c_right[1] = 1.0-(r+eps) 
            x_c_right[2] = -vx_left*d
        elseif idx == 3 # state event (bottom)
            x_d_right[1] = n_left+1
            x_c_right[3] = -1.0+(r+eps)
            x_c_right[4] = -vy_left*d
        elseif idx == 4 # state event (top)
            x_d_right[1] = n_left+1
            x_c_right[3] = 1.0-(r+eps)
            x_c_right[4] = -vy_left*d
        end

        nothing
    end

    return HUDAODEFunction(; p=p0, x_c_len=4, x_d_len=1, z_len=4, α=α, f=f, a_x=a_x, c_x=c_x)
end