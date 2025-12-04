#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module HUDADE

import SciMLSensitivity
import Random
import ComponentArrays

using PackageExtensionCompat
function __init__()
    @require_extensions
end

# Flux.jl
function FluxModel end

# FMIBase.jl
function FMUModel end

# Lux.jl
function LuxModel end
function LuxSecondOrderModel end
function LuxDiscreteModel end
function LuxNeuralODE end
function LuxSecondOrderNeuralODE end

include("const.jl")
include("event.jl")
include("solution.jl")
include("function.jl")
include("problem.jl")
include("sense.jl")

include("ForwardDiff.jl")
include("ReverseDiff.jl")

include("combination.jl")

# concepts
include("../concepts/Dense.jl")
include("../concepts/BouncingBall2D.jl")

end # module HUDADE
