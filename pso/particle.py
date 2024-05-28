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
    
    def update_velocity(self, inertia_weight, cognitive_coeff, social_coeff, global_best_position):
        """
        Func for updating velocity of particle
        Args:
            self, inertia_weight, cognitive_coeff, social_coeff, global_best_position
            The velocity can vary depending on how the user would like to allocate coeffs.
        """
        cognitive_component = cognitive_coeff * uniform(0, 1, self.position.shape) * (self.best_position - self.position)
        social_component = social_coeff * uniform(0, 1, self.position.shape) * (global_best_position - self.position)
        self.velocity = inertia_weight * self.velocity + cognitive_component + social_component

    def update_position(self):
        """
        Func for updating position of particle
            Args:
                Self.
                The function itself updates by the particle's velocity.
               """
        self.position = np.add(self.position, self.velocity).astype(np.float64)
        #store position array in type float64

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

