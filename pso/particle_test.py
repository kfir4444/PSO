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

    def test_update_velocity(self):
        self.particle_1.update_velocity(inertia_weight=1, cognitive_coeff=0, social_coeff=0, global_best_position=np.array([1, 1, 1]))
        self.assertSequenceEqual(self.particle_1.velocity.tolist(), np.array([0, 0, 1]).tolist())


    def test_update_position(self):
        self.particle_1.update_position()
        self.assertSequenceEqual(self.particle_1.position.tolist(), np.array([0, 0, 1]).tolist())
