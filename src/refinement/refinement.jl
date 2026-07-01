include("spacing.jl")
include("refinement_criteria.jl")
struct ParticleRefinement{RC, ELTYPE, SP, BARRAY, IARRAY, RB, SC}
    criteria                :: RC        # Tuple of all refinement criteria to be applied, e.g. `SpatialRefinementCriterion` and `SolutionRefinementCriterion`` 
    spacing_ratio           :: ELTYPE    # Ratio between spacings of different refinement bands, should be between 1.05 and 1.20     
    min_spacing             :: ELTYPE    # The minimum spacing being used in either boundaries or solids  
    smoothing_length_factor :: ELTYPE    # Constant corresponding to the value of smoothing length factor (= smoothing_length / particle_spacing) used in the simulation, see Eq. 35 for the merging proceduce
    splitting_pattern       :: SP
    delete_candidates       :: BARRAY  
    split_candidates        :: BARRAY  
    merge_candidates        :: IARRAY  
    resize_buffer           :: RB
    n_current_particles     :: IARRAY
    shifting_technique      :: SC
end

function ParticleRefinement(; n_particles,
                            spacing_ratio, min_spacing, resize_buffer, smoothing_length_factor = 1.2,
                            refinement_criteria=SpatialRefinementCriterion(),
                            splitting_pattern=nothing, shifting_technique=ParticleShiftingTechniqueSun2017())                
    if !(refinement_criteria isa Tuple)
        refinement_criteria = (refinement_criteria,)
    end

    ELTYPE = typeof(min_spacing)

    # Use zeros(Bool, n) instead of falses(n) for thread safety
    delete_candidates = zeros(Bool, n_particles)
    split_candidates = zeros(Bool, n_particles)
    merge_candidates = zeros(Int, n_particles)
    
    return ParticleRefinement(refinement_criteria, spacing_ratio, min_spacing, smoothing_length_factor,
                              splitting_pattern, delete_candidates, split_candidates, merge_candidates, resize_buffer, [n_particles],
                              shifting_technique)
end

# TODO
function refinement!(semi, v_ode, u_ode, v_tmp, u_tmp, integrator, t)

    foreach_system(semi) do system

        # Reset the refinement before doing anything
        reset_refinement!(system, semi)

        # Apply refinement criterion 
        apply_refinement_criteria!(system, v_ode, u_ode, semi)

        # Update the spacing of particles           (Algorithm 1)
        update_particle_spacing(system, v_ode, u_ode, semi)

        # # Split the particles                     (Algorithm 2)
        # split_particles!(system, v_ode, u_ode, semi)

        # TODO: Merge the particles                 (Algorithm 3)
        # merge_particles!(system, v_ode, u_ode, semi)

        update_nparticles_new!(system)
    end

    # Resize the semidiscretization and systems
    resize!(v_ode, u_ode, v_tmp, u_tmp, semi)

    foreach_system(semi) do system
        # TODO: Resize neighborhood search
        # resize_nhs!()

        # TODO: Update smoothing lengths
        # update_smoothing_lengths()

        # TODO: Shift the particles
        shift_particles!(system, v_ode, u_ode, semi, integrator)
        
        # TODO: Correct the particles
        # correct_particles()

    end

    return semi
end

# TODO
function create_cache_refinement(initial_condition, ::Nothing)
    return (;)
end

# TODO
# If refinement is not `Nothing` and `correction` is not `Nothing`, then throw an error
function create_cache_refinement(initial_condition, refinement)
    n_particles = nparticles(initial_condition)
    ELTYPE = eltype(initial_condition)
    NDIMS = ndims(initial_condition)

    reference_mass = zeros(ELTYPE, n_particles)
    _particle_spacing = zeros(ELTYPE, n_particles)
    is_anchor_particle = falses(n_particles)

    candidate_flags = zeros(Int, n_particles)
    candidate_offsets = zeros(Int, n_particles)

    neighbor_mass = zeros(ELTYPE, n_particles)
    neighbor_count = zeros(Int, n_particles)

    grad_density = zeros(ELTYPE, NDIMS, n_particles)
    grad_velocity = zeros(ELTYPE, NDIMS, NDIMS, n_particles)

    return (; reference_mass, _particle_spacing, is_anchor_particle,
            candidate_flags, candidate_offsets,
            neighbor_mass, neighbor_count, grad_density, grad_velocity)
end

function reset_refinement!(semi)
    foreach_system(semi) do system
        reset_refinement!(semi, system)
    end
end 

@inline reset_refinement!(system, semi) = system 

function reset_refinement!(system::AbstractFluidSystem, semi)
    return reset_refinement!(system, system.particle_refinement, semi)
end 

function reset_refinement!(system::AbstractFluidSystem, ::Nothing, semi)
    return system 
end 

function reset_refinement!(system::AbstractFluidSystem, refinement, semi)
    (; delete_candidates, split_candidates, merge_candidates, resize_buffer) = refinement

    fill!(delete_candidates, false)
    fill!(split_candidates, false)
    fill!(merge_candidates, 0)

    reset_resize_buffer!(resize_buffer, system) 
    reset_cache_refinement!(system.cache)

    return system
end   

# TODO 
function reset_cache_refinement!(cache) 
    (; reference_mass, _particle_spacing, is_anchor_particle, candidate_flags, candidate_offsets, neighbor_count, neighbor_mass) = cache 
    ELTYPE = eltype(reference_mass)

    fill!(reference_mass, zero(ELTYPE))
    fill!(_particle_spacing, zero(ELTYPE))
    fill!(is_anchor_particle, false)

    fill!(candidate_flags, 0)
    fill!(candidate_offsets, 0)

    fill!(neighbor_count, 0)
    fill!(neighbor_mass, zero(ELTYPE))
end


@inline update_smoothing_lengths!(system, v_ode, u_ode, semi) = system

@inline function update_smoothing_lengths!(system::AbstractFluidSystem, v_ode, u_ode, semi)
    return update_smoothing_lengths!(system, system.particle_refinement, v_ode, u_ode, semi)
end

@inline update_smoothing_lengths!(system::AbstractFluidSystem, ::Nothing, v_ode, u_ode, semi) = system

function update_smoothing_lengths!(system::AbstractFluidSystem, refinement, v_ode, u_ode, semi)
    (; neighbor_mass, neighbor_count, smoothing_length_factor) = refinement

    u = wrap_u(u_ode, system, semi)
    system_coords = current_coordinates(u, system)
    set_zero!(neighbor_mass_sum)
    set_zero!(neighbor_count)

    ELTYPE = eltype(u)
    inv_density = one(ELTYPE) / system.state_equation.reference_density
    inv_ndims  = one(ELTYPE) / ndims(system)

    # Calculate the total neighborhood mass around a particle 
    foreach_point_neighbor(system, system, system_coords, system_coords,
                           semi) do particle, neighbor, pos_diff, distance
        neighbor_mass[particle] += hydrodynamic_mass(system, neighbor)
        neighbor_count[particle] += 1
    end 

    # Update the smoothing length with the average neighborhood mass (Eq. 35)
    @threaded semi for particle in eachindex(neighbor_mass_sum)
        (delete_candidates[particle] || neighbor_count[particle] == 0) && return

        avg_mass = neighbor_mass[particle] * (one(ELTYPE) / neighbor_count[particle])
        new_smoothing_length = smoothing_length_factor * (inv_density * avg_mass)^inv_ndims
        set_particle_smoothing_length!(system, particle, new_smoothing_length)
    end

    return system
end

include("resize.jl")
