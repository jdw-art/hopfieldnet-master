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
        post_neuron = 1
        pattern = np.array([1, -1, 1, -1])

        dw = stdp_update(pre_neuron, post_neuron, pattern)
        self.assertAlmostEqual(dw, 0.18096748360719192, places=2)  # Adjust the expected value based on your implementation

    def test_stdp_update_weights(self):
        # Create mock input patterns
        input_patterns = np.array([[1, -1, -1, -1], [1, -1, -1, -1]])

        # Mock initial weights
        initial_weights = np.array([[0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0]])


        # Set initial weights in the network
        self.network.set_weights(initial_weights)

        num_neurons = len(input_patterns[0])
        weights = np.zeros((num_neurons, num_neurons))

        for i in range(num_neurons):
            weights[i] = stdp_update_weights(i, input_patterns, self.network)
        # Test STDP weight update
        # updated_weights = stdp_update_weights(0, input_patterns, self.network)
        # stdp_update_weights(0, input_patterns, self.network)
        # updated_weights = self.network.get_weights()

        # Assert that weights have been updated as expected
        # expected_weights = np.array([[0.0, 0.01, 0.0, 0.01],  # Adjust the expected values based on your implementation
        #                              [-0.01, 0.0, -0.01, 0.0],
        #                              [0.0, 0.01, 0.0, 0.01],
        #                              [-0.01, 0.0, -0.01, 0.0]])
        expected_weights = np.array([[0.0, 0.01809674836071919, 0.008607079764250578, 0.0],
                                    [-0.027631668254264857, 0.0, -0.01902458849001428, -0.027631668254264857],
                                    [-0.02760904260572633, -0.00951229424500714, 0.0, -0.02760904260572633],
                                    [0.0, 0.01809674836071919, 0.008607079764250578, 0.0]])
        self.assertTrue(np.array_equal(weights, expected_weights))

    def test_stdp_training(self):
        # Create a mock Hopfield network
        network = HopfieldNetwork(num_inputs=self.num_neurons, threshold=10)

        # Create mock input patterns
        input_patterns = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]])

        # Train the network using STDP
        stdp_training(network, input_patterns)

        # Get the trained weights
        trained_weights = network.get_weights()

        # Assert that weights have been updated as expected
        # expected_weights = np.array([[0.0, 0.01, 0.0, 0.01],  # Adjust the expected values based on your implementation
        #                              [-0.01, 0.0, -0.01, 0.0],
        #                              [0.0, 0.01, 0.0, 0.01],
        #                              [-0.01, 0.0, -0.01, 0.0]])
        expected_weights = np.array([[0.0, 0.2756999083856899, -0.08709900168669527, -0.4979637591103139],
        [0.8967614339338672, 0.0, -0.6784304117907765, -0.6121136882628291],
        [-0.11605039222456326, 0.9254605348518756, 0.0, 0.4690839266575313],
        [-0.4564480348159017, 0.318298585478171, -0.7025813835955068, 0.0]])

        self.assertTrue(np.array_equal(trained_weights, expected_weights))

if __name__ == '__main__':
    unittest.main()
