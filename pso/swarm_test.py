#!/usr/bin/env python3
# encoding: utf-8

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
        cls.swarm1 = Swarm(2, 3,
                           {"low" : [-1, -1, -1],
                            "high" : [1, 1, 1]},
                           {"inertia_weight" : 0.5,
                            "cognitive_coeff" : 1,
                            "social_coeff" : 1,},
                            False)

    def test_initialize_particles(self):
        self.assertEqual(self.swarm1.num_particles, len(self.swarm1.particles))

        for particle in self.swarm1.particles:
            for coord in particle.position:
                self.assertLessEqual(coord, 1)
                self.assertLessEqual(-1, coord)
    
