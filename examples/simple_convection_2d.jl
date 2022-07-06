using Pixie
using OrdinaryDiffEq

n_particles_per_dimension = (15, 3)
particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_masses = ones(Float64, prod(n_particles_per_dimension))

for y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]
    particle = (x - 1) * n_particles_per_dimension[2] + y

    # Coordinates
    particle_coordinates[1, particle] = x / 2 + 3
    particle_coordinates[2, particle] = y / 2

    # Velocity
    particle_velocities[1, particle] = -1
    particle_velocities[2, particle] = 0
end

semi = Pixie.SPHSemidiscretization{2}(particle_masses)

tspan = (0.0, 20.0)
ode = Pixie.semidiscretize(semi, particle_coordinates, particle_velocities, tspan)

alive_callback = Pixie.AliveCallback(alive_interval=100)

sol = solve(ode, RDPK3SpFSAL49(), saveat=0.2, callback=alive_callback);
