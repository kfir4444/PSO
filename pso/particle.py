from numpy import hsplit
from numpy.random import uniform
import numpy as np

class Particle:
    """
    Class Particle for swarm optimization.
    Args:
        position (np.array[float, int, np.float, np.int]): the particles initial position.
        velocity (np.array[float, int, np.float, np.int]): the particles initial velocity.
    """

    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_fitness = float('inf')

    def __str__(self):
        return f'position {self.position}, velocity {self.velocity}'
    
    def __repr__(self):
        return {"position" : self.position,
                "velocity" : self.velocity,
                "best_position" : self.best_position,
                "best_fitness" : self.best_fitness}
    
    def update_velocity_pso(self, inertia_weight, cognitive_coeff, social_coeff, global_best_position):
        """
        Func for updating velocity of particle
        Args:
            self, inertia_weight, cognitive_coeff, social_coeff, global_best_position
            The velocity can vary depending on how the user would like to allocate coeffs.
        """
        cognitive_component = cognitive_coeff * uniform(0, 1, self.position.shape) * (self.best_position - self.position)
        social_component = social_coeff * uniform(0, 1, self.position.shape) * (global_best_position - self.position)
        self.velocity = inertia_weight * self.velocity + cognitive_component + social_component

    def update_velocity_sispo(self, neighbors, kc, c1, c2, global_best_position):
        num_neighbors = len(neighbors)
        phi = c1 + c2
        eta = 2/(abs(2 - phi - np.sqrt(phi**2 - 4 * phi)))
        if num_neighbors > kc:
            neighbor_velocities = np.sum([self.velocity + np.random.uniform(0, phi) * (neighbor.best_position - self.position) for neighbor in neighbors], axis=0) / num_neighbors
            self.velocity = eta * (self.velocity + neighbor_velocities)
        else:
            cognitive_component = np.random.uniform(0, c1) * (self.best_position - self.position)
            social_component = np.random.uniform(0, c2) * (global_best_position - self.position)
            self.velocity = eta * (self.velocity + cognitive_component + social_component)

    def update_position(self):
        """
        Func for updating position of particle
            Args:
                Self.
                The function itself updates by the particle's velocity.
               """
        self.position = np.add(self.position, self.velocity).astype(np.float64)

    def evaluate_fitness(self, objective_function):
        return float(objective_function(*hsplit(self.position, self.position.shape[0])))

    def update_best_position(self, objective_function):
        """
        Func for updating best position of particle
            Args:
                self, objective function.
                The function itself updates by the particle's evaluate_fitness function for a certain objective function.
        """
        current_fitness = self.evaluate_fitness(objective_function)
        if current_fitness < self.best_fitness:
            self.best_position = np.copy(self.position)
            self.best_fitness = current_fitness
