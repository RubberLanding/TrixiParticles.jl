include("spacing.jl")
include("refinement_criteria.jl")
include("split_pattern.jl")
include("splitting.jl")
include("merge.jl")
struct ParticleRefinement{RC, ELTYPE, SP, SC}
    criteria                :: RC        # Tuple of all refinement criteria to be applied, e.g. `SpatialRefinementCriterion` and `SolutionRefinementCriterion`` 
    spacing_ratio           :: ELTYPE    # Ratio between spacings of different refinement bands, should be between 1.05 and 1.20     
    min_spacing             :: ELTYPE    # The minimum spacing being used in either boundaries or solids  
    max_spacing             :: ELTYPE    # The minimum spacing being used in either boundaries or solids  
    smoothing_length_factor :: ELTYPE    # Constant corresponding to the value of smoothing length factor (= smoothing_length / particle_spacing) used in the simulation, see Eq. 35 for the merging proceduce
    splitting_pattern       :: SP
    shifting_technique      :: SC
    reference_density       :: ELTYPE
    max_mass_factor         :: ELTYPE    # m_max = max_mass_factor * m_ref, split above and merge below this mass
    min_mass_factor         :: ELTYPE    # m_min = min_mass_factor * m_ref, only particles below this mass merge after splitting
    n_startup_rounds        :: Int       # Number of refinement rounds in `refinement_startup!`
end

function ParticleRefinement(; spacing_ratio, min_spacing, max_spacing,
                            reference_density,
                            smoothing_length_factor=1.2,
                            max_mass_factor=1.05, min_mass_factor=0.5,
                            n_startup_rounds=default_n_startup_rounds(spacing_ratio,
                                                                      min_spacing,
                                                                      max_spacing),
                            refinement_criteria=SpatialRefinementCriterion(),
                            splitting_pattern=HexagonalSplitting(),
                            shifting_technique=ParticleShiftingTechniqueSun2017())
    if !(refinement_criteria isa Tuple)
        refinement_criteria = (refinement_criteria,)
    end

    return ParticleRefinement(refinement_criteria, spacing_ratio, min_spacing, max_spacing,
                              smoothing_length_factor,
                              splitting_pattern,
                              shifting_technique, reference_density,
                              max_mass_factor, min_mass_factor, n_startup_rounds)
end

# The spacing changes by at most `spacing_ratio` per call of Alg. 1, so it takes
# `log(max_spacing / min_spacing) / log(spacing_ratio)` calls to build the full band.
# As in the reference code, use three times this number.
function default_n_startup_rounds(spacing_ratio, min_spacing, max_spacing)
    n_levels = ceil(Int, log(max_spacing / min_spacing) / log(spacing_ratio))

    return 3 * max(n_levels, 1)
end

function refinement!(semi, v_ode, u_ode, v_tmp, u_tmp, integrator, t)
    return refinement!(semi, v_ode, u_ode, v_tmp, u_tmp)
end

function refinement!(semi, v_ode, u_ode, v_tmp, u_tmp)
    foreach_system(semi) do system
        isnothing(get_refinement(system)) && return

        # Reset the refinement flags. Note that the target spacing is persistent.
        reset_refinement!(system, semi)

        # Move active particles to the start and inactive particles to the end
        # Update self-interaction nhs
        compact_system!(system, v_ode, u_ode, semi)
        neighborhood_search = get_neighborhood_search(system, semi)

        u = wrap_u(u_ode, system, semi)
        update_nhs!(neighborhood_search, system, system, u, u, semi)

        # Set the target spacing near boundaries and in criterion regions (Alg. 4)
        apply_refinement_criteria!(system, v_ode, u_ode, semi)

        # Update the spacing of all other particles (Alg. 1)
        update_particle_spacing(system, v_ode, u_ode, semi)

        # Merge before splitting, every particle below `m_max` is a candidate (Alg. 3)
        merge_particles_pre_split!(system, v_ode, u_ode, semi)
    end

    # Split the particles (Alg. 2)
    split_particles!(semi, v_ode, u_ode, v_tmp, u_tmp)

    # Merge after splitting, only particles below `m_min` are candidates (Alg. 3)
    foreach_system(semi) do system
        merge_particles!(system, v_ode, u_ode, semi)
    end

    # After adding and deleting particles,
    # we need to update the nhs
    update_nhs!(semi, u_ode)

    # Shift the particles and correct the particle properties.
    # This also updates the smoothing lengths (Eq. 35).
    foreach_system(semi) do system
        shift_particles!(system, v_ode, u_ode, semi)
    end

    update_nhs!(semi, u_ode)

    return semi
end

# Build the graded spacing field and the matching particle distribution
# before the time integration starts.
function refinement_startup!(semi, v_ode, u_ode, v_tmp, u_tmp)
    n_rounds = 0
    foreach_system(semi) do system
        refinement = get_refinement(system)
        isnothing(refinement) && return
        n_rounds = max(n_rounds, refinement.n_startup_rounds)
    end

    for _ in 1:n_rounds
        refinement!(semi, v_ode, u_ode, v_tmp, u_tmp)
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

    # Use zeros(Bool, n) instead of falses(n) for thread safety
    delete_candidates = zeros(Bool, n_particles)
    split_candidates = zeros(Bool, n_particles)
    merge_candidates = zeros(Int, n_particles)

    # Marks particles whose spacing was set by a criterion in this call.
    # Fixed particles are skipped in the spacing update (Alg. 1).
    is_fixed = zeros(Bool, n_particles)

    # The target spacing is stored per particle and persists between refinement calls.
    # Initialize it with the spacing that corresponds to the initial mass.
    _particle_spacing = (initial_condition.mass ./ refinement.reference_density) .^
                        (1 / NDIMS)
    _particle_spacing_tmp = copy(_particle_spacing)
    reference_mass = refinement.reference_density .* _particle_spacing .^ NDIMS

    candidate_flags = zeros(Int, n_particles)
    candidate_offsets = zeros(Int, n_particles)

    neighbor_mass = zeros(ELTYPE, n_particles)
    neighbor_count = zeros(Int, n_particles)

    grad_density = zeros(ELTYPE, NDIMS * n_particles)
    grad_pressure = zeros(ELTYPE, NDIMS * n_particles)
    grad_velocity = zeros(ELTYPE, NDIMS * NDIMS * n_particles)

    return (; delete_candidates, split_candidates, merge_candidates,
            is_fixed, reference_mass, _particle_spacing, _particle_spacing_tmp,
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
    reset_cache_refinement!(system, system.cache)

    return system
end

# Note that `_particle_spacing` and `reference_mass` are not reset,
# since the spacing of the previous call is needed in the spacing update (Alg. 1).
function reset_cache_refinement!(system, cache)
    (; delete_candidates, split_candidates, merge_candidates, reference_mass,
     candidate_flags, candidate_offsets, neighbor_count, neighbor_mass) = cache
    ELTYPE = eltype(reference_mass)

    fill!(delete_candidates, false)
    fill!(split_candidates, false)
    fill!(merge_candidates, 0)

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
    (; smoothing_length_factor, reference_density) = refinement
    (; neighbor_mass, neighbor_count) = system.cache

    u = wrap_u(u_ode, system, semi)
    system_coords = current_coordinates(u, system)
    set_zero!(neighbor_mass)
    set_zero!(neighbor_count)

    ELTYPE = eltype(u)
    inv_ndims = one(ELTYPE) / ndims(system)

    neighborhood_search = get_neighborhood_search(system, semi)
    backend = semi.parallelization_backend

    @threaded semi for particle in eachparticle(system)
        particle_radius = compact_support(system, system, particle)
        @inbounds foreach_neighbor(system_coords, system_coords,
                                   neighborhood_search, backend,
                                   particle;
                                   search_radius=particle_radius) do particle, neighbor, _,
                                                                     _
            neighbor_mass[particle] += hydrodynamic_mass(system, neighbor)
            neighbor_count[particle] += 1
        end
    end

    # Update the smoothing length with the average neighborhood mass (Eq. 35)
    @threaded semi for particle in eachparticle(system)
        neighbor_count[particle] == 0 && return

        avg_volume = neighbor_mass[particle] *
                     (one(ELTYPE) / (neighbor_count[particle] * reference_density))
        new_smoothing_length = smoothing_length_factor * (avg_volume)^inv_ndims
        set_particle_smoothing_length!(system, particle, new_smoothing_length)
    end

    # TrixiParticles.@autoinfiltrate

    return system
end

# A particle that is activated in a fluid system, e.g. by an inflow, gets the target spacing
# that corresponds to its mass. Otherwise, it would keep the values of the buffer slot.
@inline initialize_refined_particle!(system, particle) = system

@inline function initialize_refined_particle!(system::AbstractFluidSystem, particle)
    return initialize_refined_particle!(system, get_refinement(system), particle)
end

@inline initialize_refined_particle!(system, ::Nothing, particle) = system

@inline function initialize_refined_particle!(system, refinement, particle)
    (; _particle_spacing, reference_mass, is_fixed, delete_candidates) = system.cache
    (; reference_density) = refinement

    mass = hydrodynamic_mass(system, particle)
    _particle_spacing[particle] = (mass / reference_density)^(1 / ndims(system))
    reference_mass[particle] = mass
    is_fixed[particle] = false
    delete_candidates[particle] = false

    return system
end

@inline compact_system!(system, v_ode, u_ode, semi) = system

@inline function compact_system!(system::AbstractFluidSystem, v_ode, u_ode, semi)
    compact_system!(system, get_refinement(system), v_ode, u_ode, semi)
end

@inline compact_system!(system, refinement::Nothing, v_ode, u_ode, semi) = system

# TODO: Throw error when creating a FluidSystem with refinement but no buffer
function compact_system!(system, refinement, v_ode, u_ode, semi)
    (; buffer, cache) = system
    (; active_particle, active_particle_count, eachparticle) = buffer

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    # Sort the system based on the boolean flag
    # Active particles (true) come first, inactive particles (false) are at the end
    perm = sortperm(transfer2cpu(active_particle), rev=true)

    # Note that the following contain also inactive particles
    system_coords = current_coordinates(u, system)
    system_velocity = current_velocity(v, system)
    system_pressure = current_pressure(v, system)
    system_density = current_density(v, system)

    # Apply the compaction to standard data
    active_particle .= active_particle[perm]
    system_coords .= system_coords[:, perm]
    system_velocity .= system_velocity[:, perm]
    system_pressure .= system_pressure[perm]
    system_density .= system_density[perm]
    system.mass .= system.mass[perm]
    system.smoothing_length .= system.smoothing_length[perm]

    # Apply to data in the cache
    cache.reference_mass .= cache.reference_mass[perm]
    cache._particle_spacing .= cache._particle_spacing[perm]
    cache.is_fixed .= cache.is_fixed[perm]
    cache.delete_candidates .= cache.delete_candidates[perm]

    # Update buffer
    active_particle_count[] = count(active_particle)
    eachparticle[1:buffer.active_particle_count[]] .= 1:active_particle_count[]

    return buffer
end

# For Vectors (e.g., delta_v, grad_density, grad_pressure)
@inline function linear_index_vector(i, particle, NDIMS)
    return (particle - 1) * NDIMS + i
end

# For Matrices/Tensors (e.g., grad_velocity)
@inline function linear_index_matrix(i, j, particle, NDIMS)
    return (particle - 1) * NDIMS * NDIMS + (j - 1) * NDIMS + i
end

include("shifting.jl")
include("resize.jl") # Culprit
