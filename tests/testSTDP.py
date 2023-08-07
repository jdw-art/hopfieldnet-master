import unittest
import numpy as np
from hopfieldnet.newHopfield import HopfieldNetwork
from hopfieldnet.stdptrain import stdp_update_weights, stdp_training, stdp_update


class TestSTDPFunctions(unittest.TestCase):
    def setUp(self):
        # Create a mock Hopfield network
        self.num_neurons = 4
        self.network = HopfieldNetwork(self.num_neurons)

    def test_stdp_update(self):
        pre_neuron = 0
        post_neuron = 3
        pattern = np.array([1, -1, 1, -1])

        dw = stdp_update(pre_neuron, post_neuron, pattern)
        self.assertAlmostEqual(dw, 0.01809674836071919, places=2)  # Adjust the expected value based on your implementation

    def test_stdp_update_weights(self):
        # Create mock input patterns
        input_patterns = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]])

        # Mock initial weights
        initial_weights = np.array([[0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0]])

        # Set initial weights in the network
        self.network.set_weights(initial_weights)

        # Test STDP weight update
        updated_weights = stdp_update_weights(0, input_patterns, self.network)

        # Assert that weights have been updated as expected
        expected_weights = np.array([[0.0, 0.01, 0.0, 0.01],  # Adjust the expected values based on your implementation
                                     [-0.01, 0.0, -0.01, 0.0],
                                     [0.0, 0.01, 0.0, 0.01],
                                     [-0.01, 0.0, -0.01, 0.0]])
        self.assertTrue(np.array_equal(updated_weights, expected_weights))

    def test_stdp_training(self):
        # Create a mock Hopfield network
        network = HopfieldNetwork(num_neurons=self.num_neurons)

        # Create mock input patterns
        input_patterns = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]])

        # Train the network using STDP
        stdp_training(network, input_patterns)

        # Get the trained weights
        trained_weights = network.get_weights()

        # Assert that weights have been updated as expected
        expected_weights = np.array([[0.0, 0.01, 0.0, 0.01],  # Adjust the expected values based on your implementation
                                     [-0.01, 0.0, -0.01, 0.0],
                                     [0.0, 0.01, 0.0, 0.01],
                                     [-0.01, 0.0, -0.01, 0.0]])
        self.assertTrue(np.array_equal(trained_weights, expected_weights))

if __name__ == '__main__':
    unittest.main()
