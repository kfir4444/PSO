#!/usr/bin/env python3
# encoding: utf-8

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
        cls.particle_1 = Particle(position=np.array([0, 0 ,0]), velocity=np.array([0, 0, 1]))

    def test_update_particle_velocity(self):
        """
        Test for updating velocity of particle function
        Args:
            self.
            The function itself takes the following arguments:
            inertia_weight, cognitive_coeff, social_coeff, global_best_position
        """
        self.particle_1.update_velocity(inertia_weight=1, cognitive_coeff=0, social_coeff=0, global_best_position=np.array([1, 1, 1]))
        self.assertSequenceEqual(self.particle_1.velocity.tolist(), np.array([0, 0, 1]).tolist())


    def test_update_particle_position(self):
        """
        Test for updating position of particle function
        Args:
            self.
            The function itself updates by the particle's velocity.
        """
        self.particle_1.update_position()
        self.assertSequenceEqual(self.particle_1.position.tolist(), np.array([0, 0, 1]).tolist())

    def test_update_particle_best_position(self):
        """
        Test for updating best position of particle function
            Args:
                self.
                The function itself updates by the particle's evaluate_fitness function for a certain
                objective function.
        """
        self.particle_1.update_best_position(objective_function=lambda x, y, z: x**2 - 1)
        self.assertSequenceEqual(self.particle_1.best_position.tolist(), np.array([0, 0, 0]).tolist())
