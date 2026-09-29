include("spacing.jl")
include("refinement_criteria.jl")
include("split_pattern.jl")
include("splitting.jl")
include("merge.jl")
struct ParticleRefinement{RC, ELTYPE, SP, BARRAY, IARRAY, SC}
    criteria                :: RC        # Tuple of all refinement criteria to be applied, e.g. `SpatialRefinementCriterion` and `SolutionRefinementCriterion`` 
    spacing_ratio           :: ELTYPE    # Ratio between spacings of different refinement bands, should be between 1.05 and 1.20     
    min_spacing             :: ELTYPE    # The minimum spacing being used in either boundaries or solids  
    smoothing_length_factor :: ELTYPE    # Constant corresponding to the value of smoothing length factor (= smoothing_length / particle_spacing) used in the simulation, see Eq. 35 for the merging proceduce
    splitting_pattern       :: SP
    delete_candidates       :: BARRAY
    split_candidates        :: BARRAY
    merge_candidates        :: IARRAY
    n_current_particles     :: IARRAY
    shifting_technique      :: SC
    reference_density       :: ELTYPE
end

function ParticleRefinement(; n_particles, spacing_ratio, min_spacing,
                            reference_density,
                            smoothing_length_factor=1.2,
                            refinement_criteria=SpatialRefinementCriterion(),
                            splitting_pattern=HexagonalSplitting(),
                            shifting_technique=ParticleShiftingTechniqueSun2017())
    if !(refinement_criteria isa Tuple)
        refinement_criteria = (refinement_criteria,)
    end

    # Use zeros(Bool, n) instead of falses(n) for thread safety
    delete_candidates = zeros(Bool, n_particles)
    split_candidates = zeros(Bool, n_particles)
    merge_candidates = zeros(Int, n_particles)

    return ParticleRefinement(refinement_criteria, spacing_ratio, min_spacing,
                              smoothing_length_factor,
                              splitting_pattern, delete_candidates, split_candidates,
                              merge_candidates, [n_particles],
                              shifting_technique, reference_density)
end

# TODO
function refinement!(semi, v_ode, u_ode, v_tmp, u_tmp, integrator, t)
    foreach_system(semi) do system
        # Reset the refinement before doing anything
        reset_refinement!(system, semi)

        # Move active particles to the start and inactive particles to the end
        # Update self-interaction nhs
        compact_system!(system, v_ode, u_ode, semi)
        neighborhood_search = get_neighborhood_search(system, semi)
        update_nhs!(neighborhood_search, system, system, u_ode, u_ode, semi)

        # Apply refinement criterion 
        apply_refinement_criteria!(system, v_ode, u_ode, semi)

        # Update the spacing of particles           (Algorithm 1)
        update_particle_spacing(system, v_ode, u_ode, semi)
    end 

    # Split the particles                           (Algorithm 2)
    split_particles!(semi, v_ode, u_ode, v_tmp, u_tmp)

    foreach_system(semi) do system 
        # Merge the particles                       (Algorithm 3)
        merge_particles!(system, v_ode, u_ode, semi)

        # Move active particles to the start and inactive particles to the end
        compact_system!(system, v_ode, u_ode, semi)
    end

    # After adding and deleting particles,
    # we need to update the nhs
    update_nhs!(semi, u_ode)

    foreach_system(semi) do system
        # Update smoothing lengths
        update_smoothing_lengths!(system, v_ode, u_ode, semi)

        # Shift the particles and correct the particle properties
        shift_particles!(system, v_ode, u_ode, semi, integrator.dt)
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

    candidate_flags = zeros(Int, n_particles)
    candidate_offsets = zeros(Int, n_particles)

    neighbor_mass = zeros(ELTYPE, n_particles)
    neighbor_count = zeros(Int, n_particles)

    grad_density = zeros(ELTYPE, NDIMS * n_particles)
    grad_pressure = zeros(ELTYPE, NDIMS * n_particles)
    grad_velocity = zeros(ELTYPE, NDIMS * NDIMS * n_particles)

    return (; reference_mass, _particle_spacing,
            candidate_flags, candidate_offsets,
            neighbor_mass, neighbor_count, grad_density, grad_pressure, grad_velocity)
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
    (; delete_candidates, split_candidates, merge_candidates) = refinement

    fill!(delete_candidates, false)
    fill!(split_candidates, false)
    fill!(merge_candidates, 0)

    reset_cache_refinement!(system, system.cache)

    return system
end

# TODO 
function reset_cache_refinement!(system, cache)
    (; particle_spacing) = system.initial_condition
    (; reference_mass, _particle_spacing, candidate_flags, candidate_offsets,
     neighbor_count, neighbor_mass) = cache
    ELTYPE = eltype(reference_mass)

    fill!(reference_mass, zero(ELTYPE))
    fill!(_particle_spacing, particle_spacing)

    fill!(candidate_flags, 0)
    fill!(candidate_offsets, 0)

    fill!(neighbor_count, 0)
    fill!(neighbor_mass, zero(ELTYPE))
end

@inline update_smoothing_lengths!(system, v_ode, u_ode, semi) = system

@inline function update_smoothing_lengths!(system::AbstractFluidSystem, v_ode, u_ode, semi)
    return update_smoothing_lengths!(system, system.particle_refinement, v_ode, u_ode, semi)
end

@inline update_smoothing_lengths!(system::AbstractFluidSystem, ::Nothing, v_ode, u_ode,
                                  semi) = system

function update_smoothing_lengths!(system::AbstractFluidSystem, refinement, v_ode, u_ode,
                                   semi)
    (; smoothing_length_factor, delete_candidates, reference_density) = refinement
    (; neighbor_mass, neighbor_count) = system.cache

    u = wrap_u(u_ode, system, semi)
    v = wrap_v(v_ode, system, semi)

    system_coords = current_coordinates(u, system)
    set_zero!(neighbor_mass)
    set_zero!(neighbor_count)

    ELTYPE = eltype(u)
    inv_ndims = one(ELTYPE) / ndims(system)

    # Calculate the total neighborhood mass around a particle 
    foreach_point_neighbor(system, system, system_coords, system_coords,
                           semi) do particle, neighbor, pos_diff, distance
        neighbor_mass[particle] += hydrodynamic_mass(system, neighbor)
        neighbor_count[particle] += 1
    end

    # TrixiParticles.@autoinfiltrate

    # Update the smoothing length with the average neighborhood mass (Eq. 35)
    @threaded semi for particle in eachindex(neighbor_mass)
        neighbor_count[particle] == 0 && return

        avg_volume = neighbor_mass[particle] *
                     (one(ELTYPE) / (neighbor_count[particle] * reference_density))
        new_smoothing_length = smoothing_length_factor * (avg_volume)^inv_ndims
        set_particle_smoothing_length!(system, particle, new_smoothing_length)
    end

    # TrixiParticles.@autoinfiltrate

    return system
end

@inline compact_system!(system, v_ode, u_ode, semi) = system

@inline function compact_system!(system::AbstractFluidSystem, v_ode, u_ode, semi)
    compact_system!(system, get_refinement(system), v_ode, u_ode, semi)
end 

@inline compact_system!(system, refinement::Nothing, v_ode, u_ode, semi) = system

# TODO: Throw error when creating a FluidSystem with refinement but no buffer
function compact_system!(system, refinement, v_ode, u_ode, semi)
    (; buffer) = system
    (; active_particle, active_particle_count, eachparticle) = buffer

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    # Sort the system based on the boolean flag
    # Active particles (true) come first, inactive particles (false) are at the end
    compaction_perm = sortperm(transfer2cpu(active_particle), rev=true)

    # Note that the following contain also inactive particles
    system_coords = current_coordinates(u, system)
    system_velocity = current_velocity(v, system)
    system_pressure = current_pressure(v, system)
    system_density = current_density(v, system)

    # Apply the compaction to standard data
    active_particle .= active_particle[compaction_perm]
    system_coords .= system_coords[:, compaction_perm]
    system_velocity .= system_velocity[:, compaction_perm]
    system_pressure .= system_pressure[compaction_perm]
    system_density .= system_density[compaction_perm]
    system.mass .= system.mass[compaction_perm]
    system.smoothing_length .= system.smoothing_length[compaction_perm]

    # Update buffer
    active_particle_count[] = count(active_particle)
    eachparticle[1:buffer.active_particle_count[]] .= 1:active_particle_count[]

    return buffer
end

include("shifting.jl")
include("resize.jl")
