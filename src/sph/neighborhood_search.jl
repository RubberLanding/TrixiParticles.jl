@inline initialize!(neighborhood_search, u, semi; particles=nothing) = nothing
@inline update!(neighborhood_search, u, semi) = nothing

# Without neighborhood search iterate over all particles
@inline eachneighbor(particle, u, neighborhood_search::Nothing, semi; particles=eachparticle(semi)) = particles


@doc raw"""
    SpatialHashingSearch{NDIMS}(search_radius)

Simple neighborhood search with uniform search radius ``d`` based on (Ihmsen et al. 2011, Section 4.4).

The domain is divided into cells of uniform size ``d`` in each dimension.
Only particles in neighboring cells are then considered as neighbors of a particle.
Instead of representing a finite domain by an array of cells (basic uniform grid),
a potentially infinite domain is represented by saving cells in a hash table,
indexed by the cell index tuple
```math
\left( \left\lfloor \frac{x}{d} \right\rfloor, \left\lfloor \frac{y}{d} \right\rfloor \right) \quad \text{or} \quad
\left( \left\lfloor \frac{x}{d} \right\rfloor, \left\lfloor \frac{y}{d} \right\rfloor, \left\lfloor \frac{z}{d} \right\rfloor \right),
```
where ``x, y, z`` are the space coordinates.

As opposed to (Ihmsen et al. 2011), we do not handle the hashing explicitly and use
Julia's `Dict` data structure instead.

References:
- Markus Ihmsen, Nadir Akinci, Markus Becker, Matthias Teschner.
  "A Parallel SPH Implementation on Multi-Core CPUs".
  In: Computer Graphics Forum 30.1 (2011), pages 99–112.
  [doi: 10.1111/J.1467-8659.2010.01832.X](https://doi.org/10.1111/J.1467-8659.2010.01832.X)
"""
struct SpatialHashingSearch{NDIMS, ELTYPE}
    hashtable       ::Dict{NTuple{NDIMS, Int64}, Vector{Int64}}
    search_radius   ::ELTYPE
    empty_vector    ::Vector{Int64} # Just an empty vector

    function SpatialHashingSearch{NDIMS}(search_radius) where NDIMS
        hashtable = Dict{NTuple{NDIMS, Int64}, Vector{Int64}}()
        empty_vector = Vector{Int64}()

        new{NDIMS, typeof(search_radius)}(hashtable, search_radius, empty_vector)
    end
end


function initialize!(neighborhood_search::SpatialHashingSearch, u, semi; particles=eachparticle(semi))
    @unpack hashtable = neighborhood_search

    empty!(hashtable)

    # This is needed to prevent lagging on macOS ARM.
    # See https://github.com/JuliaSIMD/Polyester.jl/issues/89
    ThreadingUtilities.sleep_all_tasks()

    for particle in particles
        # Get cell index of the particle's cell
        cell_coords = get_cell_coords(u, neighborhood_search, semi, particle)

        # Add particle to corresponding cell or create cell if it does not exist
        if haskey(hashtable, cell_coords)
            append!(hashtable[cell_coords], particle)
        else
            hashtable[cell_coords] = [particle]
        end
    end

    return neighborhood_search
end


# Modify the existing hash table by moving particles into their new cells
function update!(neighborhood_search::SpatialHashingSearch, u, semi)
    @unpack hashtable = neighborhood_search

    # This is needed to prevent lagging on macOS ARM.
    # See https://github.com/JuliaSIMD/Polyester.jl/issues/89
    ThreadingUtilities.sleep_all_tasks()

    for (cell_coords, particles) in hashtable
        # Find all particles whose coordinates do not match this cell
        moved_particle_indices = (i for i in eachindex(particles)
                                  if get_cell_coords(u, neighborhood_search, semi, particles[i]) != cell_coords)

        # Add moved particles to new cell
        for i in moved_particle_indices
            particle = particles[i]
            new_cell_coords = get_cell_coords(u, neighborhood_search, semi, particle)

            # Add particle to corresponding cell or create cell if it does not exist
            if haskey(hashtable, new_cell_coords)
                append!(hashtable[new_cell_coords], particle)
            else
                hashtable[new_cell_coords] = [particle]
            end
        end

        # Remove moved particles from this cell or delete the cell if it is now empty
        if count(_ -> true, moved_particle_indices) == length(particles)
            delete!(hashtable, cell_coords)
        else
            deleteat!(particles, moved_particle_indices)
        end
    end

    return neighborhood_search
end


@inline function eachneighbor(particle, u, neighborhood_search::SpatialHashingSearch{2}, semi; particles=nothing)
    cell_coords = get_cell_coords(u, neighborhood_search, semi, particle)
    x, y = cell_coords
    # Generator of all neighboring cells to consider
    neighboring_cells = ((x + i, y + j) for i = -1:1, j = -1:1)

    # Merge all lists of particles in the neighboring cells into one iterator
    Iterators.flatten(particles_in_cell(cell, neighborhood_search) for cell in neighboring_cells)
end

@inline function eachneighbor(particle, u, neighborhood_search::SpatialHashingSearch{3}, semi; particles=nothing)
    cell_coords = get_cell_coords(u, neighborhood_search, semi, particle)
    x, y, z = cell_coords
    # Generator of all neighboring cells to consider
    neighboring_cells = ((x + i, y + j, z + k) for i = -1:1, j = -1:1, k = -1:1)

    # Merge all lists of particles in the neighboring cells into one iterator
    Iterators.flatten(particles_in_cell(cell, neighborhood_search) for cell in neighboring_cells)
end


@inline function particles_in_cell(cell_index, neighborhood_search)
    @unpack hashtable, empty_vector = neighborhood_search

    if haskey(hashtable, cell_index)
        return hashtable[cell_index]
    end

    # Reuse empty vector to avoid allocations
    return empty_vector
end


@inline function get_cell_coords(u, neighborhood_search, semi, particle)
    @unpack search_radius = neighborhood_search

    return Tuple(floor.(Int64, get_particle_coords(u, semi, particle) / search_radius))
end