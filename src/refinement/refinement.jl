include("spacing.jl")
include("refinement_criteria.jl")
struct ParticleRefinement{RC, ELTYPE, SP, BARRAY, IARRAY, RB}
    criteria                :: RC        # Tuple of all refinement criteria to be applied, e.g. `SpatialRefinementCriterion` and `SolutionRefinementCriterion`` 
    max_spacing_ratio       :: ELTYPE    # Ratio between spacings of different refinement bands, should be between 1.05 and 1.20     
    min_spacing             :: ELTYPE    # The minimum spacing being used in either boundaries or solids  
    smoothing_length_factor :: ELTYPE    # Constant corresponding to the value of smoothing length factor (= smoothing_length / particle_spacing) used in the simulation, see Eq. 35 for the merging proceduce
    splitting_pattern       :: SP
    delete_candidates       :: BARRAY  
    split_candidates        :: BARRAY  
    merge_candidates        :: IARRAY  
    candidate_flags         :: IARRAY
    candidate_offsets       :: IARRAY
    resize_buffer           :: RB
    n_current_particles     :: IARRAY
end

function ParticleRefinement(; n_particles, smoothing_length, initial_particle_spacing,
                            max_spacing_ratio, min_spacing, resize_buffer, smoothing_length_factor = 1.2,
                            refinement_criteria=SpatialRefinementCriterion(),
                            splitting_pattern=nothing)
    if !(refinement_criteria isa Tuple)
        refinement_criteria = (refinement_criteria,)
    end

    delete_candidates = Vector{Bool}(undef, n_particles)
    split_candidates = Vector{Bool}(undef, n_particles)
    merge_candidates = Vector{Int}(undef, n_particles)

    candidate_flags = Vector{Int}(undef, n_particles)
    candidate_offsets = Vector{Int}(undef, n_particles)

    return ParticleRefinement(refinement_criteria, max_spacing_ratio, min_spacing, smoothing_length_factor,
                              splitting_pattern, delete_candidates, split_candidates, merge_candidates, candidate_flags, candidate_offsets, 
                              resize_buffer, [n_particles])
end

# TODO
function refinement!(semi, v_ode, u_ode, v_tmp, u_tmp, t)

    foreach_system(semi) do system

        # Reset the refinement before doing anything
        reset_refinement!(system, semi)

        # Apply refinement criterion 
        apply_refinement_criteria!(system, v_ode, u_ode, semi)

        # Update the spacing of particles           (Algorithm 1)
        update_particle_spacing(system, v_ode, u_ode, semi)

        # Split the particles                       (Algorithm 2)
        split_particles!(system, v_ode, u_ode, semi)

        # TODO: Merge the particles                 (Algorithm 3)
        # for _ in 1:3 
        #     merge_particles!(system, v_ode, u_ode, semi)
        # end

        update_nparticles_new!(system)
    end

    # # Resize the semidiscretization
    # resize!(semi, v_ode, u_ode, v_tmp, u_tmp)

    foreach_system(semi) do system

        # # Resize the systems
        # resize!(system, v_ode, u_ode, semi) 

        # TODO: Resize neighborhood search
        # resize_nhs!()

        # TODO: Shift the particles
        # for _ in 1:3
        #     shift_particles!()
        # end 
        
        # TODO: Correct the particles
        # correct_particles()

        # TODO: Update smoothing lengths
        # update_smoothing_lengths()
    end

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

    reference_mass = zeros(ELTYPE, n_particles)
    _particle_spacing = zeros(ELTYPE, n_particles)
    is_anchor_particle = falses(n_particles)

    return (; reference_mass, _particle_spacing, is_anchor_particle)
end

function reset_refinement!(semi)
    foreach_system(semi) do system
        reset_refinement!(semi, system)
    end
end 

function reset_refinement!(system, semi) 
    return system
end 

function reset_refinement!(system::AbstractFluidSystem, semi)
    return reset_refinement!(system, system.particle_refinement, semi)
end 

function reset_refinement!(system::AbstractFluidSystem, ::Nothing, semi)
    return system 
end 

function reset_refinement!(system::AbstractFluidSystem, refinement, semi)
    (; delete_candidates, split_candidates, merge_candidates, candidate_flags, candidate_offsets, resize_buffer) = refinement

    fill!(delete_candidates, false)
    fill!(split_candidates, false)
    fill!(merge_candidates, false)
    fill!(candidate_flags, 0)
    fill!(candidate_offsets, 0) 

    reset_resize_buffer!(resize_buffer, system) 
end   

# TODO 
function reset_cache_refinement!(cache) end
