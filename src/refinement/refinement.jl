struct ParticleRefinement{RC, ELTYPE, SP}
    criteria            :: RC             # Tuple of all refinement criteria to be applied, e.g. `SpatialRefinementCriterion` and `SolutionRefinementCriterion`` 
    max_spacing_ratio   :: ELTYPE         # Ratio between spacings of different refinement bands, should be between 1.05 and 1.20
    min_spacing         :: ELTYPE         # The minimum spacing being used in either boundaries or solids
    splitting_pattern   :: SP
    split_candidates    :: Vector{Int}
    n_new_particles     :: Ref{Int}
    n_current_particles :: Ref{Int}
end

function ParticleRefinement(; n_particles, smoothing_length, initial_particle_spacing,
                            max_spacing_ratio, min_spacing,
                            refinement_criteria=nothing,
                            splitting_pattern=nothing)
    if !(refinement_criteria isa Tuple)
        refinement_criteria = (refinement_criteria,)
    end

    return ParticleRefinement(refinement_criteria, max_spacing_ratio, min_spacing,
                              splitting_pattern, Int[], Ref(0), Ref(0))
end

# TODO:
function refinement!(semi, v_ode, u_ode, v_tmp, u_tmp, t)
    # Apply refinement criterion, e.g. for SpatialRefinementCriterion setting the spacing of particles near the boundary 
    # apply_refinement_criteria!()

    # Update the spacing of particles (Algorthm 1)
    # update_particle_spacing()

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
function create_cache_refinement(initial_condition, ::Nothing, smoothing_length)
    return (;)
end

# TODO
# If refinement is not `Nothing` and `correction` is not `Nothing`, then throw an error
function create_cache_refinement(initial_condition, refinement, smoothing_length)
    return (;)
end

# TODO 
function reset_cache_refinement!(cache) end
