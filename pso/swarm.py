from numpy import abs, array, zeros, meshgrid, linspace
from numpy.random import uniform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from particle import Particle

class Swarm:
    """
    Class Swarm for swarm optimization.
    Args:
        num_particles (int): the number of particles that make up the swarm.
        num_dimensions (int): the number of dimensions in the search space. This has to match the number of dimensions
                                in the objective function we are optimizing.
        search_range (dict): A dictionary specifying the lower and upper bounds of the search space for each dimension.
                             Example: {"low": [lower_bound_1, lower_bound_2, ..., lower_bound_n],
                                       "high": [upper_bound_1, upper_bound_2, ..., upper_bound_n]}
        params (dict): A dictionary of parameters for the optimization algorithm. The parameters are different depending
                        what optimization method the user chooses.
                       For PSO: {"inertia_weight": float, "cognitive_coeff": float, "social_coeff": float}
                       For SISPO: {"c1": float, "c2": float, "kc": int}
        plot (bool): Whether to plot the optimization process. Only relevant for up to 2D.
        optimization_type (str): The type of optimization algorithm to use. Options are 'pso' (default) and 'sispo'.
    """
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

    def generate_random_position(self):
        """
         Func for generating random positions for each particle in the swarm within the defined search range.
         Args:
             self
             The particles' initial position is limited to the search_range argument.
         """
        return uniform(self.search_range['low'], self.search_range['high'], self.num_dimensions)

    def generate_random_velocity(self):
        """
         Func for generating random velocity for each particle in the swarm within the defined search range.
         Args:
             self
             The particles' initial velocity is limited to the search_range argument.
         """
        return uniform(-abs(array(self.search_range['low']) - array(self.search_range['high'])),
                        abs(array(self.search_range['low']) - array(self.search_range['high'])),
                        self.num_dimensions)

    def initialize_particles(self):
        """
        Func for initializing the particles' position and velocity.
        Args:
            self
            The particles initial position is limited to the search_range argument. The amount of particles is dictated
            by num_particles argument.
        """
        particles = []
        for _ in range(self.num_particles):
            position = self.generate_random_position()
            velocity = self.generate_random_velocity()
            particle = Particle(position, velocity)
            particles.append(particle)
        return particles

    def prepare_plot(self, objective_function, center=None):
        """
        Func that prepares the plot for visualization  of swarm optimization.
        Args:
            self, objective_function, center (None is default)
            The plot is centered in the middle of the search range defined, however we usually like to show the plot
            centered around the minimum of the objective function which have defined in the plotter function.
        """
        if center is None:
            low, high = self.search_range['low'], self.search_range['high']
        else:
            low = [center[0] - 10, center[1] - 10]
            high = [center[0] + 10, center[1] + 10]

        self.x = linspace(low[0], high[0], 1000)
        self.y = linspace(low[1], high[1], 1000)
        self.z = zeros((len(self.x), len(self.y)))

        for i in range(len(self.x)):
            for j in range(len(self.y)):
                self.z[j, i] = objective_function(self.x[i], self.y[j])

    def plotter(self, objective_function):
        """
        Func that plots the swarm optimization of a specific objective function.
        Args:
            self, objective_function
            The center is chosen to be the min position of the objective function so the convergence is shown in a
            manner that is easier to see.
        """
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

    def determine_neighbors(self):
        """
        Func that determines what particles and how many particles are considered neighbors for each individual particle.
        Args:
            self
            Neighbors are particles that are closer than the mean euclidean distance from another particle. We might
            consider a different definition of neighbors in the future.
        """
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
        """
        Func that updates the global best position based on the best fitness score of all the particles in the swarm.
        Args:
            self
            Each particle has a different fitness score during each iteration of the optimization process, and the best
            global position is updated to the position at which a particle receives a better fitness score than the
            previous global best.
        """
        for particle in self.particles:
            if particle.best_fitness < self.global_best_fitness:
                self.global_best_position = particle.best_position
                self.global_best_fitness = particle.best_fitness

    def optimize(self, objective_function, max_iterations):
        """
        Func that executes the swarm optimization of a specific objective function.
        Args:
            self, objective_function, max_iterations
            This function implements all prior functions and executes the whole optimization process given a swarm and
            objective function.
        """
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
        