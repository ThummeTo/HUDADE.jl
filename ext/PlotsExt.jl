#
# Copyright (c) 2024 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module PlotsExt

using HUDADE, Plots
import HUDADE: unsense

"""
    Plots.plot(solution::HUDADESolution; plotkwargs...)

Plots the `solution` of a simulation and returns the corresponding figure.
(requires package Plots.jl)

# Arguments
- `solution::HUDADESolution`: Struct containing information about the solution values, success, states and events of a specific FMU simulation.

# Keywords
- `plotkwargs...`: Arguments, that are passed on to Plots.plot!
"""
function Plots.plot(solution::HUDADESolution; plotkwargs...)
    fig = Plots.plot(; xlabel = "t [s]")
    Plots.plot!(fig, solution; plotkwargs...)
    return fig
end

"""
    Plots.plot!(fig::Plots.Plot, solution::HUDADESolution; 
                states::Union{Bool, Nothing}=nothing,
                values::Union{Bool, Nothing}=nothing,
                stateEvents::Union{Bool, Nothing}=nothing,
                timeEvents::Union{Bool, Nothing}=nothing,
                stateIndices=nothing,
                valueIndices=nothing,
                maxLabelLength=64,
                plotkwargs...)

Plots the `solution` of a FMU simulation into `fig` and returns the figure again.

# Arguments
- `fig::Plots.Plot`: Figure to plot into
- `solution::FMUSolution`: Struct containing information about the solution values, success, states and events of a specific FMU simulation.
- `states::Union{Bool, Nothing}=nothing`: controls if states should be plotted (default = nothing: plot states from `solution`, if they exist)
- `values::Union{Bool, Nothing}=nothing`: controls if values should be plotted (default = nothing: plot values from `solution`, if they exist)
- `stateEvents::Union{Bool, Nothing}=nothing`: controls if stateEvents should be plotted (default = nothing: plot stateEvents from `solution`, if at least one and at most 100 exist)
- `timeEvents::Union{Bool, Nothing}=nothing`: controls if timeEvents should be plotted (default = nothing: plot timeEvents from `solution`, if at least one and at most 100 exist)
- `stateIndices=nothing`: controls which states will be plotted by index in state vector (default = nothing: plot all states)
- `valueIndices=nothing`: controls which values will be plotted by index (default = nothing: plot all values)
- `maxLabelLength::Integer=64`: controls the maximum length for legend labels (too long labels are cut from front)
- `maxStateEvents::Integer=100`: controls, how many state events are plotted before suppressing plotting state events
- `maxTimeEvents::Integer=100`: controls, how many time events are plotted before suppressing plotting state events
- `plotkwargs...`: Arguments, that are passed on to Plots.plot!
"""
function Plots.plot!(
    fig::Plots.Plot,
    solution::HUDADESolution;
    states::Union{Bool,Nothing} = nothing,
    values::Union{Bool,Nothing} = nothing,
    stateEvents::Union{Bool,Nothing} = nothing,
    timeEvents::Union{Bool,Nothing} = nothing,
    stateIndices = nothing,
    valueIndices = nothing,
    maxLabelLength::Integer = 64,
    maxStateEvents::Integer = 100,
    maxTimeEvents::Integer = 100,
    plotkwargs...,
)
    numStates = 0
    numStateEvents = 0
    numTimeEvents = 0
    for e in solution.events
        if e.idx > 0
            numStateEvents += 1
        else
            numTimeEvents += 1
        end
    end

    if isnothing(states)
        states = !isnothing(solution.u) && length(solution.u) > 0
    end

    if states 
        numStates = length(solution.u[1])
    end

    if isnothing(values)
        # ToDo: values = !isnothing(solution.saveval) && length(solution.saveval) > 0
        values = false
    end

    if isnothing(stateEvents)
        stateEvents = false
        for e in solution.events
            if e.idx > 0
                stateEvents = true
                break
            end
        end

        if numStateEvents > maxStateEvents
            @info "plot(...): Number of state events ($(numStateEvents)) exceeding 100, disabling automatic plotting of state events (can be forced with keyword `stateEvents=true`)."
            stateEvents = false
        end
    end

    if isnothing(timeEvents)
        timeEvents = false
        for e in solution.events
            if e.idx == 0
                timeEvents = true
                break
            end
        end

        if numTimeEvents > maxTimeEvents
            @info "plot(...): Number of time events ($(numTimeEvents)) exceeding 100, disabling automatic plotting of time events (can be forced with keyword `timeEvents=true`)."
            timeEvents = false
        end
    end

    if isnothing(stateIndices)
        stateIndices = 1:numStates
    end

    # if isnothing(valueIndices)
    #     if !isnothing(solution.saveval)
    #         valueIndices = 1:length(solution.saveval.saveval[1])
    #     end
    # end

    plot_min = Inf
    plot_max = -Inf

    # plot states
    if states
        t = collect(unsense(e) for e in solution.t)
        numValues = length(solution.u[1])

        for v = 1:numValues
            if v ∈ stateIndices
                vals = collect(unsense(data[v]) for data in solution.u)

                plot_min = min(plot_min, vals...)
                plot_max = max(plot_max, vals...)

                # prevent legend labels from getting too long
                label = "$(v)"
                labelLength = length(label)
                if labelLength > maxLabelLength
                    label = "..." * label[labelLength-maxLabelLength:end]
                end

                Plots.plot!(fig, t, vals; label = label, plotkwargs...)
            end
        end
    end

    # plot recorded values
    if values
        t = collect(unsense(e) for e in solution.saveval.t)
        numValues = length(solution.saveval.saveval[1])

        for v = 1:numValues
            if v ∈ valueIndices
                vr = "[unknown]"
                vrName = "[unknown]"
                if !isnothing(solution.valueReferences) &&
                   v <= length(solution.valueReferences)
                    vr = solution.valueReferences[v]
                    vrNames = valueReferenceToString(instance.fmu, vr)
                    vrName = length(vrNames) > 0 ? vrNames[1] : "?"
                end

                vals = collect(unsense(data[v]) for data in solution.saveval.saveval)

                plot_min = min(plot_min, vals...)
                plot_max = max(plot_max, vals...)

                # prevent legend labels from getting too long
                label = "$vrName ($vr)"
                labelLength = length(label)
                if labelLength > maxLabelLength
                    label = "..." * label[labelLength-maxLabelLength:end]
                end

                Plots.plot!(fig, t, vals; label = label, plotkwargs...)
            end
        end
    end

    if stateEvents
        first = true
        for e in solution.events
            if e.idx > 0
                Plots.plot!(
                    fig,
                    [e.t, e.t],
                    [plot_min, plot_max];
                    label = (first ? "State event(s)" : nothing),
                    style = :dash,
                    color = :blue,
                )
                first = false
            end
        end
    end

    if timeEvents
        first = true
        for e in solution.events
            if e.idx == 0
                Plots.plot!(
                    fig,
                    [e.t, e.t],
                    [plot_min, plot_max];
                    label = (first ? "Time event(s)" : nothing),
                    style = :dash,
                    color = :red,
                )
                first = false
            end
        end
    end

    return fig
end

end # PlotsExt.jl
