from numpy import abs, array, zeros, meshgrid, linspace
from numpy.random import uniform
import matplotlib.pyplot as plt
from matplotlib import cm
from particle import Particle

class Swarm:
    def __init__(self, num_particles, num_dimensions, search_range, params, plot):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.search_range = search_range
        self.global_best_position = zeros(num_dimensions)
        self.inertia_weight = params["inertia_weight"]
        self.cognitive_coeff = params["cognitive_coeff"]
        self.social_coeff = params["social_coeff"]
        self.plot = plot
        self.global_best_fitness = float('inf')
        
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
    
    def prepare_plot(self, objective_function):
        if any(map(lambda x: x is None, [self.x, self.y, self.z])):
            low, high = self.search_range['low'], self.search_range['high']
            self.x = linspace(low[0], high[0], 1000)
            self.y = linspace(low[1], high[1], 1000)
            self.z = zeros(shape=(self.x.shape[0], self.y.shape[0]))
            for indexi, i in enumerate(self.x):
                for indexj, j in enumerate(self.y):
                    self.z[indexi, indexj] = objective_function(i, j)

    def plotter(self, objective_function):
        self.prepare_plot(objective_function)
        X, Y = meshgrid(self.x, self.y)
        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, self.z, cmap=cm.PuBu_r)
        cbar = fig.colorbar(cs)
        p = array([particle.position for particle in self.particles])
        v = array([particle.velocity for particle in self.particles])
        ax.scatter(p[:, 0], p[:, 1], c="k", marker='+')
        plt.show()

    def generate_random_position(self):
        return uniform(self.search_range['low'], self.search_range['high'], self.num_dimensions)

    def generate_random_velocity(self):
        return uniform(-abs(array(self.search_range['low']) - array(self.search_range['high'])),
                        abs(array(self.search_range['low']) - array(self.search_range['high'])),
                        self.num_dimensions)

    def update_global_best_position(self):
        for particle in self.particles:
            if particle.best_fitness < self.global_best_fitness:
                self.global_best_position = particle.best_position
                self.global_best_fitness = particle.best_fitness

    def optimize(self, objective_function, max_iterations):
        iter = 0
        if self.plot:
            self.plotter(objective_function)
        while iter < max_iterations:
            iter += 1
            self.update_global_best_position()

            # Update each particle's velocity and position
            for particle in self.particles:
                particle.update_velocity(self.inertia_weight,
                                         self.cognitive_coeff,
                                         self.social_coeff,
                                         global_best_position=self.global_best_position)
                particle.update_position()

                # Evaluate fitness and update personal best position
                particle.evaluate_fitness(objective_function)
                particle.update_best_position(objective_function)
            if self.plot:
                self.plotter(objective_function)

        return self.global_best_position, self.global_best_fitness
                




class SwarmError(Exception):
    def __init__(self, message):            
        super().__init__(message)
        