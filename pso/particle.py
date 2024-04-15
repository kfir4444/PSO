from numpy import hsplit
from numpy.random import uniform

class Particle:
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
        cognitive_component = cognitive_coeff * uniform(0, 1, self.position.shape) * (self.best_position - self.position)
        social_component = social_coeff * uniform(0, 1, self.position.shape) * (global_best_position - self.position)
        self.velocity = inertia_weight * self.velocity + cognitive_component + social_component

    def update_position(self):
        self.position += self.velocity

    def evaluate_fitness(self, objective_function):
        return objective_function(*hsplit(self.position, self.position.shape[0]))

    def update_best_position(self, objective_function):
        current_fitness = self.evaluate_fitness(objective_function)
        if current_fitness < self.best_fitness:
            self.best_position = self.position
            self.best_fitness = current_fitness
