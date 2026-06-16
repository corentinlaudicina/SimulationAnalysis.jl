
module SimulationAnalysis
import Base.show, Quickhull
using HDF5, Polyester, Tullio, LoopVectorization, Base.Threads, Parsers, DelimitedFiles, Random, IfElse, Dierckx, ProgressMeter, OffsetArrays, ChunkSplitters, CellListMap, StaticArrays, Bessels
using LinearAlgebra

export SingleComponentSimulation, MultiComponentSimulation, SelfPropelledVoronoiSimulation, MCSPVSimulation
export KSpace, construct_k_space
export find_density_modes, SingleComponentDensityModes, MultiComponentDensityModes
export find_intermediate_scattering_function, find_self_intermediate_scattering_function
export find_structure_factor
export find_mean_squared_displacement, find_non_gaussian_parameter
export find_overlap_function
export find_relative_distance_neighborlists, find_absolute_distance_neighborlists, find_voronoi_neighborlists
export find_CB
export chi4_per_simulation, compute_G4_per_simulation
export _gaussian_mobilities, _heaviside_mobilities, _bond_breaking_mobilities, _corrected_gaussian_mobilities
export find_relaxation_time
export find_radial_distribution_function

abstract type Simulation end
import Base.step

include("Simulation.jl")
include("LoadData.jl")
include("Kspace.jl")
include("CorrelationFunction.jl")
include("DensityModes.jl")
include("IntermediateScatteringFunction.jl")
include("StructureFactors.jl")
include("Forces.jl")
include("RadialDistributionFunction.jl")
include("MeanSquaredDisplacement.jl")
include("F4_diagonal.jl")
include("OverlapFunction.jl")
include("Neighborlists.jl")
include("BondBreakingParameter.jl")
include("OverlapFluctuationFunction.jl")
include("Clustering.jl")
include("Utils.jl")
include("CurrentModes.jl")
include("VelocityCorrelations.jl")
include("COMCorrection.jl")


function show(io::IO,  ::MIME"text/plain", s::Union{KSpace, Simulation})
    println(io, "This is a $(typeof(s)).")
    println(io, "It contains the fields: ")
    for fieldname in fieldnames(typeof(s))
        if getfield(s, fieldname) isa Union{Int, Float64, String}
            println(io, "$(fieldname): $(getfield(s, fieldname))")
        else
            println(io, "$(fieldname): $(typeof(getfield(s, fieldname)))")
        end
    end
end

end # module

