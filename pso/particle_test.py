#!/usr/bin/env python3
# encoding: utf-8

from copy import deepcopy
import unittest
import numpy as np

from particle import Particle


class Testparticle(unittest.TestCase):
    """

    """

    @classmethod
    def setUpClass(cls):
        """

        """
        cls.particle_1 = Particle(position=np.array([0, 0, 0]), velocity=np.array([0, 0, 1]))
        cls.particle_2 = Particle(position=np.array([1]), velocity=np.array([-1]))

    def test_update_particle_velocity(self):
        particle_1 = deepcopy(self.particle_1)
        particle_1.update_velocity(inertia_weight=1,
                                   cognitive_coeff=0,
                                   social_coeff=0,
                                   global_best_position=np.array([1, 1, 1])
                                   )
        self.assertSequenceEqual(self.particle_1.velocity.tolist(), np.array([0, 0, 1]).tolist())

    def test_update_particle_position(self):
        particle_1 = deepcopy(self.particle_1)
        particle_1.update_position()
        self.assertSequenceEqual(self.particle_1.position.tolist(), np.array([0, 0, 0]).tolist())

    def test_update_particle_best_position(self):
        particle_1 = deepcopy(self.particle_1)
        particle_1.position = np.array([0.25, 0, 0])
        particle_1.update_best_position(objective_function=lambda x, y, z: x ** 2 - 2 * x - 1)
        print("particle_1 best fitness before update", particle_1.best_fitness)
        print("particle_1 fitness before position update:",particle_1.evaluate_fitness(lambda x, y, z: x ** 2 - 2 * x - 1))
        particle_1.velocity = np.array([-0.25, 0, 0])
        particle_1.update_position()
        print("particle_1 position and velocity:", particle_1)
        print("particle_1 fitness after position update:", particle_1.evaluate_fitness(lambda x, y, z: x ** 2 - 2 * x - 1))
        print("particle_1 best fitness after update",particle_1.best_fitness)
        particle_1.update_best_position(objective_function=lambda x, y, z: x ** 2 - 2*x - 1)
        self.assertSequenceEqual(particle_1.best_position.tolist(), np.array([0.25, 0, 0]).tolist())

