from numpy import abs, array, zeros, meshgrid, linspace
from numpy.random import uniform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from particle import Particle

class Swarm:
    def __init__(self, num_particles, num_dimensions, search_range, params, plot, optimization_type='pso'):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.search_range = search_range
        self.global_best_position = zeros(num_dimensions)
        self.plot = plot
        self.global_best_fitness = float('inf')
        self.optimization_type = optimization_type

        if optimization_type == 'pso':
            self.inertia_weight = params["inertia_weight"]
            self.cognitive_coeff = params["cognitive_coeff"]
            self.social_coeff = params["social_coeff"]
            self.c1 = 0  # Not used in PSO
            self.c2 = 0  # Not used in PSO
            self.kc = 0  # Not used in PSO
        elif optimization_type == 'sispo':
            self.c1 = params["c1"]
            self.c2 = params["c2"]
            self.kc = params["kc"]
            self.inertia_weight = 0  # Not used in SISPO
            self.cognitive_coeff = 0  # Not used in SISPO
            self.social_coeff = 0  # Not used in SISPO
        self.particles = self.initialize_particles()        
        if self.plot:
            self.x, self.y, self.z = None, None, None

    def __str__(self):
        return "\n".join([p.__str__() for p in self.particles])

    def __repr__(self):
        return "\n".join([p.__repr__() for p in self.particles])
    
    def initialize_particles(self):
        particles = []
        for _ in range(self.num_particles):
            position = self.generate_random_position()
            velocity = self.generate_random_velocity()
            particle = Particle(position, velocity)
            particles.append(particle)
        return particles

    def prepare_plot(self, objective_function, center=None):
        if center is None:
            low, high = self.search_range['low'], self.search_range['high']
        else:
            low = [center[0] - 2, center[1] - 2]
            high = [center[0] + 2, center[1] + 2]

        self.x = linspace(low[0], high[0], 1000)
        self.y = linspace(low[1], high[1], 1000)
        self.z = zeros((len(self.x), len(self.y)))

        for i in range(len(self.x)):
            for j in range(len(self.y)):
                self.z[j, i] = objective_function(self.x[i], self.y[j])

    def plotter(self, objective_function):
        min_position = self.global_best_position
        self.prepare_plot(objective_function, center=min_position)
        X, Y = np.meshgrid(self.x, self.y)
        fig, ax = plt.subplots()
        min_value = np.min(self.z)
        max_value = np.max(self.z)
        contour_levels = np.linspace(min_value, max_value, 10)
        cs = ax.contourf(X, Y, self.z, levels=contour_levels, cmap=cm.PuBu_r)
        cbar = fig.colorbar(cs)
        p = np.array([particle.position for particle in self.particles])
        ax.scatter(p[:, 0], p[:, 1], c="k", marker='+')
        plt.show()

    def generate_random_position(self):
        return uniform(self.search_range['low'], self.search_range['high'], self.num_dimensions)

    def generate_random_velocity(self):
        return uniform(-abs(array(self.search_range['low']) - array(self.search_range['high'])),
                        abs(array(self.search_range['low']) - array(self.search_range['high'])),
                        self.num_dimensions)

    def determine_neighbors(self):
        distances = np.zeros((self.num_particles, self.num_particles))
        neighbors = [[] for _ in range(self.num_particles)]

        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                dist = np.linalg.norm(self.particles[i].position - self.particles[j].position)
                distances[i][j] = dist
                distances[j][i] = dist

        avg_distance = np.mean(distances)

        for i in range(self.num_particles):
            for j in range(self.num_particles):
                if distances[i][j] <= avg_distance and i != j:
                    neighbors[i].append(self.particles[j])

        return neighbors, distances

    def update_global_best_position(self):
        for particle in self.particles:
            if particle.best_fitness < self.global_best_fitness:
                self.global_best_position = particle.best_position
                self.global_best_fitness = particle.best_fitness

    def optimize(self, objective_function, max_iterations):
        for particle in self.particles:
            particle.best_fitness = particle.evaluate_fitness(objective_function)
            particle.best_position = particle.position

        self.update_global_best_position()

        for iteration in range(max_iterations):
            neighbors, distances = self.determine_neighbors()

            for i, particle in enumerate(self.particles):
                if self.optimization_type == 'pso':
                    particle.update_velocity_pso(self.inertia_weight, self.cognitive_coeff, self.social_coeff,
                                                 self.global_best_position)
                elif self.optimization_type == 'sispo':
                    particle.update_velocity_sispo(neighbors[i], self.kc, self.c1, self.c2,
                                                   self.global_best_position)
                particle.update_position()
                particle.evaluate_fitness(objective_function)
                particle.update_best_position(objective_function)

            self.update_global_best_position()

            if self.plot:
                self.plotter(objective_function)

        return self.global_best_position, self.global_best_fitness


class SwarmError(Exception):
    def __init__(self, message):            
        super().__init__(message)
        