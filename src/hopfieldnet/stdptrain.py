import numpy as np


def stdp_update_weights(neuron_index, input_patterns, network):
    """Update the weights for the given neuron using STDP"""
    num_patterns = len(input_patterns)
    num_neurons = len(input_patterns[0])

    weights = np.zeros(num_neurons)

    for j in range(num_neurons):
        if neuron_index == j: continue

        dw = 0.0
        for mu in range(num_patterns):
            dw += stdp_update(neuron_index, j, input_patterns[mu])

        weights[j] = network.get_weights()[neuron_index][j] + dw

    return weights


def stdp_update(pre_neuron, post_neuron, pattern):
    """Calculate the STDP update for a specific pre-synaptic and post-synaptic neuron"""
    tau_pos = 20  # Time constant for potentiation
    tau_neg = 20  # Time constant for depression
    A_pos = 0.01  # Learning rate for potentiation
    A_neg = -0.01  # Learning rate for depression

    dw = 0.0

    for t in range(len(pattern)):
        if pattern[pre_neuron] == 1 and pattern[post_neuron] == -1:  # Pre-synaptic spike, post-synaptic no spike
            # dt = t - np.where(pattern[:t][::-1] == -1)[0][0]  # Calculate the time difference to the last post-synaptic spike
            # dw += A_pos * np.exp(-dt / tau_pos)
            last_post_spike = np.where(pattern[:t][::-1] == -1)[0]
            if len(last_post_spike) > 0:
                dt = t - last_post_spike[0]
                dw += A_pos * np.exp(-dt / tau_neg)

        if pattern[pre_neuron] == -1 and pattern[post_neuron] == 1:  # Pre-synaptic no spike, post-synaptic spike
            # dt = t - np.where(pattern[:t][::-1] == 1)[0][0]  # Calculate the time difference to the last pre-synaptic spike
            # dw += A_neg * np.exp(-dt / tau_neg)
            last_post_spike = np.where(pattern[:t][::-1] == 1)[0]
            if len(last_post_spike) > 0:
                dt = t - last_post_spike[0]
                dw += A_pos * np.exp(-dt / tau_neg)

    return dw


def stdp_training(network, input_patterns):
    """Train a network using the STDP learning rule"""
    num_neurons = len(input_patterns[0])

    weights = np.zeros((num_neurons, num_neurons))

    for i in range(num_neurons):
        weights[i] = stdp_update_weights(i, input_patterns, network)

    network.set_weights(weights)
