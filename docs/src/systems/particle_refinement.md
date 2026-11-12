# [Particle Refinement](@id particle_refinement)

Adaptive particle refinement changes the resolution of a fluid during the simulation,
following [Haftu et al. (2022)](@cite Haftu2022).
Particles are split where a finer resolution is needed, for example near boundaries,
and merged where a coarser resolution is sufficient.
Every particle has its own mass and smoothing length.

Refinement is enabled by passing a [`ParticleRefinement`](@ref) as `particle_refinement`
to an [`EntropicallyDampedSPHSystem`](@ref) or a [`WeaklyCompressibleSPHSystem`](@ref)
and adding a [`ParticleRefinementCallback`](@ref) to the callbacks.
Since splitting adds particles, the fluid system needs a `buffer_size`.

```julia
refinement = ParticleRefinement(; spacing_ratio=1.2, min_spacing=particle_spacing / 2,
                                max_spacing=particle_spacing,
                                reference_density=fluid_density)

fluid_system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                           sound_speed, particle_refinement=refinement,
                                           buffer_size=1000)

# ...

callbacks = CallbackSet(info_callback, saving_callback,
                        ParticleRefinementCallback(interval=10))
```

See `examples/fluid/taylor_green_vortex_2d_refinement.jl` and
`examples/fluid/pipe_flow_2d_refinement.jl` for complete examples.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("refinement", "refinement.jl")]
```

## Refinement Criteria

The refinement criteria set the target spacing of the particles in some regions.
The target spacing of all other particles is graded from there.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("refinement", "refinement_criteria.jl"),
         joinpath("refinement", "spacing.jl")]
```

## Splitting and Merging

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("refinement", "split_pattern.jl"),
         joinpath("refinement", "splitting.jl"),
         joinpath("refinement", "merging.jl"),
         joinpath("refinement", "resize.jl")]
```

## Shifting

After splitting and merging, the particles are shifted to a more regular distribution.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("refinement", "shifting.jl")]
```
