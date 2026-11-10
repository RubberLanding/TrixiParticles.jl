@testset verbose=true "Refinement" begin
    include("spacing.jl")
    include("refinement_criteria.jl")
    include("variable_smoothing_length.jl")
    include("interactions.jl")
    include("edac.jl")
end
