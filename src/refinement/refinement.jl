include("spacing.jl")
include("refinement_criteria.jl")
struct ParticleRefinement{RC, ELTYPE, SP, BARRAY, IARRAY, RB}
    criteria                :: RC        # Tuple of all refinement criteria to be applied, e.g. `SpatialRefinementCriterion` and `SolutionRefinementCriterion`` 
    max_spacing_ratio       :: ELTYPE    # Ratio between spacings of different refinement bands, should be between 1.05 and 1.20     
    min_spacing             :: ELTYPE    # The minimum spacing being used in either boundaries or solids  
    smoothing_length_factor :: ELTYPE    # Constant corresponding to the value of smoothing length factor (= smoothing_length / particle_spacing) used in the simulation, see Eq. 35 for the merging proceduce
    splitting_pattern       :: SP
    delete_candidates       :: BARRAY  
    split_candidates        :: IARRAY  
    merge_candidates        :: IARRAY  
    n_new_particles         :: IARRAY 
    n_current_particles     :: IARRAY
    resize_buffer           :: RB
end

function ParticleRefinement(; n_particles, smoothing_length, initial_particle_spacing,
                            max_spacing_ratio, min_spacing, resize_buffer, smoothing_length_factor = 1.2,
                            refinement_criteria=SpatialRefinementCriterion(),
                            splitting_pattern=nothing)
    if !(refinement_criteria isa Tuple)
        refinement_criteria = (refinement_criteria,)
    end

    delete_candidates = Vector{Bool}(undef, n_particles)
    split_candidates = Vector{Int}(undef, n_particles)
    merge_candidates = Vector{Int}(undef, n_particles)

    n_new_particles = zeros(Int, 1)
    n_current_particles = zeros(Int, 1)

    return ParticleRefinement(refinement_criteria, max_spacing_ratio, min_spacing, smoothing_length_factor,
                              splitting_pattern, delete_candidates, split_candidates, merge_candidates, 
                              n_new_particles, n_current_particles, resize_buffer)
end

# TODO:
function refinement!(semi, v_ode, u_ode, v_tmp, u_tmp, t)
    # Apply refinement criterion, e.g. for SpatialRefinementCriterion setting the spacing of particles near the boundary 
    apply_refinement_criteria!(semi, v_ode, u_ode)

    # Update the spacing of particles (Algorthm 1)
    update_particle_spacing(semi, v_ode, u_ode)

    # Split the particles (Algorithm 2)
    # split_particles!()

    # Merge the particles (Algorithm 3)
    # merge_particles!()

    # Shift the particles
    # shift_particles!()

    # Correct the particles
    # correct_particles()

    # Update smoothing lengths
    # update_smoothing_lengths()

    # Resize neighborhood search
    # resize_nhs!()

    return semi
end

# TODO
function create_cache_refinement(initial_condition, ::Nothing, initial_smoothing_length)
    return (;)
end

# TODO
# If refinement is not `Nothing` and `correction` is not `Nothing`, then throw an error
function create_cache_refinement(initial_condition, refinement, initial_smoothing_length)
    n_particles = length(initial_condition.mass)
    ELTYPE = eltype(initial_condition)

    reference_mass = Vector{ELTYPE}(undef, n_particles)
    _particle_spacing = Vector{ELTYPE}(undef, n_particles)

    return (; reference_mass, _particle_spacing)
end

# TODO 
function reset_cache_refinement!(cache) end
