import numpy as np
from random import randint, shuffle
from matplotlib import pyplot as plt
import matplotlib.cm as cm


class LIFNeuron:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.threshold = 0.7
        self.potential = 0.0
        self.fired = False
        self.degradation = 0.9

    def calculate_potential(self, inputs, weight):
        self.potential = (self.potential + np.sum(np.multiply(inputs, weight))) * self.degradation
        return self.potential

    def fire(self, potential):
        self.fired = potential >= self.threshold
        return 1.0 if self.fired else 0.0

    def reset(self):
        self.potential = 0.0
        self.fired = False


class InvalidWeightsException(Exception):
    pass


class InvalidNetworkInputException(Exception):
    pass


class HopfieldNetwork(object):
    def __init__(self, num_inputs, threshold=0):
        self._num_inputs = num_inputs
        self.threshold = threshold
        self.neurons = [LIFNeuron(num_inputs) for _ in range(num_inputs)]
        self._weights = np.random.uniform(0.0, 1.0, (num_inputs, num_inputs))

    def set_weights(self, weights):
        """Update the weights array"""
        if weights.shape != (self._num_inputs, self._num_inputs):
            raise InvalidWeightsException()

        self._weights = weights

    def get_weights(self):
        """Return the weights array"""
        return self._weights

    def calculate_neuron_output(self, neuron, input_pattern):
        """Calculate the output of the given neuron"""
        # for i in range(len(input_pattern)):
        #     self.neurons[neuron].potential = self.neurons[neuron].calculate_potential(input_pattern, self._weights[neuron][i])
        total_potential = self.neurons[neuron].calculate_potential(input_pattern, self._weights[neuron])
        neuron_output = self.neurons[neuron].fire(total_potential)
        return neuron_output

    # def run_once(self, update_list, input_pattern):
    #     """Iterate over every neuron and update it's output"""
    #     result = input_pattern.copy()
    #     e = self.energy(result)
    #
    #     changed = False
    #     for neuron in update_list:
    #         neuron_output = self.calculate_neuron_output(neuron, result)
    #         new_energy = self.energy(neuron_output)
    #
    #         if e == new_energy:
    #             changed = False
    #         else:
    #             result[neuron] = neuron_output
    #             changed = True
    #
    #         e = new_energy
    #
    #     return changed, result

    def run_once(self, update_list, input_pattern):
        """Iterate over every neuron and update it's output"""
        result = input_pattern.copy()

        changed = False
        for neuron in update_list:
            neuron_output = self.calculate_neuron_output(neuron, result)

            if neuron_output != result[neuron]:
                result[neuron] = neuron_output
                changed = True

        return changed, result

    def run(self, input_pattern, max_iterations=10):
        """Run the network using the input data until the output state doesn't change
        or a maximum number of iteration has been reached."""
        iteration_count = 0

        result = input_pattern.copy()

        while True:
            update_list = list(range(self._num_inputs))
            shuffle(update_list)

            changed, result = self.run_once(update_list, result)

            iteration_count += 1

            if not changed or iteration_count == max_iterations:
                return result

    def energy(self, s):
        s_matrix = np.outer(s, s)
        energy = -0.5 * np.sum(s_matrix * self._weights) + np.sum(s * self.threshold)
        return energy
        # return -0.5 * s @ self._weights @ s + np.sum(s * self.threshold)

    def plot_weights(self):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self._weights, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.savefig("weights.png")
        plt.show()

