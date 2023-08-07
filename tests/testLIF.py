import unittest

import numpy as np


class LIFNeuron:
    def __init__(self, threshold):
        self.threshold = threshold
        self.potential = 0.0
        self.fired = False
        self.degradation = 0.9

    def calculate_potential(self, inputs, weight):
        self.potential = (self.potential + np.sum(np.multiply(inputs, weight))) * self.degradation
        return self.potential

    def fire(self):
        self.fired = self.potential >= self.threshold
        return 1.0 if self.fired else 0.0

    def reset(self):
        self.potential = 0.0
        self.fired = False




class TestLIFNeuron(unittest.TestCase):
    def test_calculate_potential(self):
        neuron = LIFNeuron(threshold=1.0)
        inputs = [0.5, 0.2, 0.7]
        weight = [0.1, 0.2, 0.3]
        potential = neuron.calculate_potential(inputs, weight)
        self.assertAlmostEqual(potential, 0.27, places=2)

    def test_fire_below_threshold(self):
        neuron = LIFNeuron(threshold=1.0)
        neuron.potential = 0.8
        output = neuron.fire()
        self.assertEqual(output, 0.0)
        self.assertFalse(neuron.fired)

    def test_fire_above_threshold(self):
        neuron = LIFNeuron(threshold=1.0)
        neuron.potential = 1.2
        output = neuron.fire()
        self.assertEqual(output, 1.0)
        self.assertTrue(neuron.fired)

    def test_reset(self):
        neuron = LIFNeuron(threshold=1.0)
        neuron.potential = 0.8
        neuron.fired = True
        neuron.reset()
        self.assertEqual(neuron.potential, 0.0)
        self.assertFalse(neuron.fired)

if __name__ == '__main__':
    unittest.main()
