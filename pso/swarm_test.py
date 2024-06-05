#!/usr/bin/env python3
# encoding: utf-8
import copy
from copy import deepcopy
import unittest
import numpy as np

from particle import Particle
from swarm import Swarm


class TestSwarm(unittest.TestCase):
    """

    """

    @classmethod
    def setUpClass(cls):
        """

        """
        cls.swarm1 = Swarm(5, 3,
                           {"low" : [-1, -1, -1],
                            "high" : [1, 1, 1]},
                           {"inertia_weight" : 0.5,
                            "cognitive_coeff" : 1,
                            "social_coeff" : 1,},
                            False)
        cls.swarm2 = Swarm(50, 3,
                           {"low": [-100, -100, -100],
                            "high": [100, 100, 100]},
                           {"inertia_weight": 0.5,
                            "cognitive_coeff": 1,
                            "social_coeff": 1, },
                           False)
        cls.swarm3 = Swarm(3, 1,
                           {"low": [-1],
                            "high": [1]},
                           {"inertia_weight": 1,
                            "cognitive_coeff": 1,
                            "social_coeff": 1, },
                           False)
    def test_initialize_particles(self):
        self.assertEqual(self.swarm1.num_particles, len(self.swarm1.particles))

        for particle in self.swarm1.particles:
            for coord in particle.position:
                self.assertLessEqual(coord, 1)
                self.assertLessEqual(-1, coord)

    def test_initialize_particles2(self):
        self.assertEqual(self.swarm2.num_particles, len(self.swarm2.particles))

        for particle in self.swarm2.particles:
            for coord in particle.position:
                self.assertLessEqual(coord, 100)
                self.assertLessEqual(-100, coord)

    def test_generate_random_position(self):
        for par in self.swarm2.particles:
            for coordinate in par.position:
                self.assertLessEqual(coordinate, 100)
                self.assertLessEqual(-100, coordinate)

    def test_generate_random_velocity(self):
        for par in self.swarm1.particles:
            for coordinate in par.velocity:
                self.assertLessEqual(coordinate, 2)
                self.assertLessEqual(-2, coordinate)


    def test_update_global_best_position(self):
        particles = [
                Particle(np.array([1, 2]), np.array([-1, -1])),
                Particle(np.array([2, 3]), np.array([0, 0])),
                Particle(np.array([4, 5]), np.array([1, 1]))
            ]
        swarm4 = Swarm(3, 2,
                           {"low": [-5, -5],
                            "high": [5, 5]},
                           {"inertia_weight": 1,
                            "cognitive_coeff": 0,
                            "social_coeff": 0, },
                           False)
        swarm4.particles = particles

        for p in swarm4.particles:
            p.best_fitness = p.evaluate_fitness(objective_function=lambda x, y: x**2 +y**2)

        # Update the global best position
        swarm4.update_global_best_position()
        self.assertEqual(swarm4.global_best_fitness, 5)
        self.assertEqual(swarm4.global_best_position.tolist(), np.array([1, 2]).tolist())

    def test_optimize(self):
        # Define the objective function
        objective_function = lambda x, y, z: x**2 + y**2 + z**2

        # Run the optimize method
        max_iterations = 1000
        best_position, best_fitness = self.swarm2.optimize(objective_function, max_iterations)

        # Check if the global best position is close to minimum of the function
        np.testing.assert_array_almost_equal(best_position, [0, 0, 0], decimal=1)

        # Check if the global best fitness is close to 0 (minimum fitness)
        self.assertAlmostEqual(best_fitness, 0, places=1)
        print("best position is:",best_position,"best fitness is:", best_fitness)


