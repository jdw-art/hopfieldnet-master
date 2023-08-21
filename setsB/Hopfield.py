import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm


class LIFNeuron:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.threshold = 0.5
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

    # 使用Hebb学习规则调整连接权重
    def stdp_update_weights(self, neuron_index, input_patterns, network):
        """Update the weights for the given neuron using STDP"""
        num_patterns = len(input_patterns)
        num_neurons = len(input_patterns[0])

        weights = np.zeros(num_neurons)

        for j in range(num_neurons):
            if neuron_index == j: continue

            dw = 0.0
            for mu in range(num_patterns):
                dw += self.stdp_update(neuron_index, j, input_patterns[mu])

            weights[j] = network.get_weights()[neuron_index][j] + dw

        return weights

    def stdp_update(self, pre_neuron, post_neuron, pattern):
        """Calculate the STDP update for a specific pre-synaptic and post-synaptic neuron"""
        tau_pos = 20  # Time constant for potentiation
        tau_neg = 20  # Time constant for depression
        A_pos = 0.01  # Learning rate for potentiation
        A_neg = -0.01  # Learning rate for depression

        dw = 0.0

        for t in range(len(pattern)):
            if pattern[pre_neuron] == 1 and pattern[post_neuron] == -1:  # Pre-synaptic spike, post-synaptic no spike
                # Calculate the time difference to the last post-synaptic spike
                # dt = t - np.where(pattern[:t][::-1] == -1)[0][0]
                # dw += A_pos * np.exp(-dt / tau_pos)
                last_post_spike = np.where(pattern[:t][::-1] == -1)[0]
                if len(last_post_spike) > 0:
                    dt = t - last_post_spike[0]
                    dw += A_pos * np.exp(-dt / tau_pos)

            if pattern[pre_neuron] == -1 and pattern[post_neuron] == 1:  # Pre-synaptic no spike, post-synaptic spike
                # Calculate the time difference to the last pre-synaptic spike
                # dt = t - np.where(pattern[:t][::-1] == 1)[0][0]
                # dw += A_neg * np.exp(-dt / tau_neg)
                last_post_spike = np.where(pattern[:t][::-1] == 1)[0]
                if len(last_post_spike) > 0:
                    dt = t - last_post_spike[0]
                    dw += A_neg * np.exp(-dt / tau_neg)

        return dw

    def stdp_training(self, network, input_patterns):
        """Train a network using the STDP learning rule"""
        num_neurons = len(input_patterns[0])

        weights = np.zeros((num_neurons, num_neurons))

        for i in range(num_neurons):
            weights[i] = self.stdp_update_weights(i, input_patterns, network)

        network.set_weights(weights)

    def predict(self, data, num_iter=20, threshold=0):
        print("Start to predict...")
        self.num_iter = num_iter
        self.threshold = threshold

        # Copy to avoid call by reference
        copied_data = np.copy(data)

        # Define predict list
        predicted = []
        for i in tqdm(range(len(data))):
            predicted.append(self._run(copied_data[i]))
        return predicted

    # 对吸引子网络中的状态进行更新和迭代，init_s表示出事的神经元状态向量
    def _run(self, init_s):
        # 同步更新神经元节点
        # 初始化神经元状态 s 为 init_s。
        # 计算初始状态的能量 e，并进行迭代更新。
        # 在每次迭代中，通过计算连接权重矩阵 self.W 与神经元状态 s 的乘积，然后与阈值向量 self.threshold 进行比较，得到新的神经元状态 s。
        # 计算新的状态能量 e_new，如果当前状态的能量与新的状态能量相等（即状态收敛到吸引子），则返回最终状态 s，否则继续进行下一次迭代，直到达到最大迭代次数
        """
                    Synchronous update
                    """
        # Compute initial state energy 计算初神经元状态
        s = init_s

        e = self.energy(s)

        # Iteration
        for i in range(self.num_iter):
            # Update s
            s = np.sign(self.W @ s - self.threshold)
            # Compute new state energy
            e_new = self.energy(s)

            # 当前状态已经收敛到吸引子
            # s is converged
            if e == e_new:
                return s
            # Update energy
            e = e_new
        return s

    # 吸引子网络的能量函数
    def energy(self, s):
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)

    def plot_weights(self):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.savefig("weights.png")
        plt.show()