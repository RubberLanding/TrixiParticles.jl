using TrixiParticles
using OrdinaryDiffEqCore
using OrdinaryDiffEqLowStorageRK

@testset "Integration Test: Splitting and Merging" begin
    # ==========================================================================================
    # ==== Resolution
    particle_spacing = 0.025

    # ==========================================================================================
    # ==== Experiment Setup
    reynolds_number = 100.0

    box_length = 1.0

    U = 1.0 # m/s
    fluid_density = 1.0
    sound_speed = 10U

    b = -8pi^2 / reynolds_number

    n_particles_xy = round(Int, box_length / particle_spacing)

    # ==========================================================================================
    # ==== Fluid
    nu = U * box_length / reynolds_number

    background_pressure = sound_speed^2 * fluid_density
    shifting_technique = TransportVelocityAdami(; background_pressure)

    smoothing_length = 1.2 * particle_spacing
    smoothing_kernel = SchoenbergQuinticSplineKernel{2}()

    fluid = RectangularShape(particle_spacing, (n_particles_xy, n_particles_xy), (0.0, 0.0), density=fluid_density)

    n_particles = TrixiParticles.nparticles(fluid)
    buffer = ResizeBuffer(fluid)
    refinement = ParticleRefinement(n_particles=n_particles,
                                    spacing_ratio=1.05,
                                    min_spacing=particle_spacing/4,
                                    resize_buffer=buffer,
                                    reference_density=fluid_density,
                                    refinement_criteria=SolutionRefinementCriterion())

    viscosity = ViscosityAdami(; nu)
    density_calculator = SummationDensity()
    correction = nothing

    idx = 55
    idxs = [34, 35, 36, 37,
            44, 45, 46, 47,
            54, 55, 56, 57,
            64, 65, 66, 67,
            ]
    fluid_system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                            sound_speed, density_calculator,
                                            shifting_technique,
                                            viscosity, 
                                            correction,
                                            pressure_acceleration=nothing,
                                            particle_refinement=refinement)
    fluid_system.cache.density .= fluid_density

    # ==========================================================================================
    # ==== Test
    periodic_box = PeriodicBox(min_corner=[0.0, 0.0], max_corner=[box_length, box_length])
    semi = Semidiscretization(fluid_system,
                            neighborhood_search=TrivialNeighborhoodSearch{2}(; periodic_box))

    ode = semidiscretize(semi, (0.0, 1.0))

    v_ode, u_ode = ode.u0.x
    u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
    v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)

    # fluid_coords = TrixiParticles.current_coordinates(u, fluid_system)
    # fluid_nhs = TrixiParticles.get_neighborhood_search(fluid_system, semi)
    
    # neighbor_particles = []
    # PointNeighbors.foreach_neighbor(fluid_coords, fluid_coords, fluid_nhs, idx) do particle, neighbor, pos_diff, distance
    #     push!(neighbor_particles, neighbor)
    # end

    # TrixiParticles.@autoinfiltrate

    # Allocate temporary backup arrays of the exact same type and size
    v_tmp = similar(v_ode)
    u_tmp = similar(u_ode)

    # TrixiParticles.@autoinfiltrate
    # Reset the refinement before doing anything
    TrixiParticles.reset_refinement!(fluid_system, semi)

    # Trigger splitting and merging explicitly 
    # fluid_system.cache.reference_mass .= (fluid.mass[1] / 1.05) + 0.001
    # fluid_system.cache.reference_mass[idxs] .= (fluid.mass[1] / 1.05) - 0.001

    TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)
    TrixiParticles.update_particle_spacing(fluid_system, v_ode, u_ode, semi)

    # TrixiParticles.@autoinfiltrate
    TrixiParticles.split_particles!(fluid_system, v_ode, u_ode, semi)
    TrixiParticles.@autoinfiltrate
    TrixiParticles.update_nparticles_new!(fluid_system)
    TrixiParticles.resize!(v_ode, u_ode, v_tmp, u_tmp, semi)

    # TrixiParticles.@autoinfiltrate
    TrixiParticles.reset_resize_buffer!(fluid_system.particle_refinement.resize_buffer, fluid_system)
    TrixiParticles.merge_particles!(fluid_system, v_ode, u_ode, semi)
    TrixiParticles.@autoinfiltrate
    TrixiParticles.update_nparticles_new!(fluid_system)
    TrixiParticles.resize!(v_ode, u_ode, v_tmp, u_tmp, semi)

    u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
    ps_before_update = [TrixiParticles.particle_spacing(fluid_system, particle) for particle in TrixiParticles.eachparticle(fluid_system)]
    ic_before_update = InitialCondition(coordinates=u, density=1000.0, mass=fluid_system.smoothing_length, pressure=ps_before_update)
    TrixiParticles.trixi2vtk(ic_before_update, filename="ic_before_update")

    TrixiParticles.update_smoothing_lengths!(fluid_system, v_ode, u_ode, semi)

    u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
    ps_after_update = [TrixiParticles.particle_spacing(fluid_system, particle) for particle in TrixiParticles.eachparticle(fluid_system)]
    ic_after_update = InitialCondition(coordinates=u, density=1000.0, mass=fluid_system.smoothing_length, pressure=ps_after_update)
    TrixiParticles.trixi2vtk(ic_after_update, filename="ic_after_update")

    TrixiParticles.shift_particles!(fluid_system, v_ode, u_ode, semi, 0.01)

    u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
    ps_after_shift = [TrixiParticles.particle_spacing(fluid_system, particle) for particle in TrixiParticles.eachparticle(fluid_system)]
    ic_after_shift = InitialCondition(coordinates=u, density=1000.0, mass=fluid_system.smoothing_length, pressure=ps_after_shift)
    TrixiParticles.trixi2vtk(ic_after_shift, filename="ic_after_shift")

    TrixiParticles.@autoinfiltrate
end 

