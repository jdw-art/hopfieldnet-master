from random import randint

import numpy as np
from matplotlib import pyplot as plt

# from hopfieldnet.net import HopfieldNetwork
# from hopfieldnet.trainers import hebbian_training
from hopfieldnet.newHopfield import HopfieldNetwork
# from hopfieldnet.stdpnet import HopfieldNetwork
from hopfieldnet.stdptrain import stdp_training

import torch
import random
from torchvision import transforms, datasets

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='E:\Dataset\MNIST', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

test_dataset = datasets.MNIST(root='E:\Dataset\MNIST', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

def salt_pepper_noise(image, ratio):
    image = reshape(image)
    output = np.zeros(image.shape, np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            rand = random.random()
            if rand < ratio:  # salt pepper noise
                if random.random() > 0.5:  # change the pixel to 255
                    output[i][j] = 1.0
                else:
                    output[i][j] = 0
            else:
                output[i][j] = image[i][j]

    output = output.astype(np.float64)

    return output.reshape(784)

def binarize_array(input_array):
    # 计算数组中的最小值和最大值
    min_value = np.min(input_array)
    max_value = np.max(input_array)

    # 将数组中的值线性映射到0-1的区间，并进行四舍五入得到二值数组
    binary_array = np.round((input_array - min_value) / (max_value - min_value))

    return binary_array

def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

def plot(feature, predicted, tag1, tag2):
    feature = [reshape(data) for data in feature]
    predicted = [reshape(data) for data in predicted]
    fig, axs = plt.subplots(4, 5, figsize=(12, 8))
    fig.suptitle(tag1 + ' and ' + tag2 + ' Images', fontsize=16)

    for i in range(5):
        # Plot feature images
        axs[0, i].imshow(feature[i], cmap='gray')
        axs[0, i].set_title(tag1 + f' {i}')
        axs[0, i].axis('off')

        # Plot predicted images
        axs[1, i].imshow(predicted[i], cmap='gray')
        axs[1, i].set_title(tag2 + f' {i}')
        axs[1, i].axis('off')

    for i in range(5, 10):
        # Plot feature images
        axs[2, i-5].imshow(feature[i], cmap='gray')
        axs[2, i-5].set_title(tag1 + f' {i}')
        axs[2, i-5].axis('off')

        # Plot predicted images
        axs[3, i-5].imshow(predicted[i], cmap='gray')
        axs[3, i-5].set_title(tag2 + f' {i}')
        axs[3, i-5].axis('off')

    plt.tight_layout()
    plt.show()

def plot1(feature, predicted, tag1, tag2):
    feature = [reshape(data) for data in feature]
    predicted = [reshape(data) for data in predicted]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(tag1 + ' and ' + tag2 + ' Images', fontsize=16)

    for i in range(3):
        # Plot feature images
        axs[0, i].imshow(feature[i], cmap='gray')
        axs[0, i].set_title(tag1 + f' {i}')
        axs[0, i].axis('off')

        # Plot predicted images
        axs[1, i].imshow(predicted[i], cmap='gray')
        axs[1, i].set_title(tag2 + f' {i}')
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()

test_sets = []
test_sets1 = []
# 存储每个类别对应的图像
images_per_class = {i: [] for i in range(10)}
# 遍历测试集，将每个类别的图像存储到相应的列表中
for images, labels in test_loader:
    image = images[0][0]  # 由于batch_size为1，所以取第一个图像
    label = labels[0].item()
    images_per_class[label].append(image)

for i in range(3):
    random_image = random.choice(images_per_class[i])
    test_sets.append(np.array(random_image).reshape(784))

for i in range(3):
    random_image = random.choice(images_per_class[i])
    test_sets1.append(np.array(random_image).reshape(784))

original_sets =[binarize_array(data) for data in test_sets]
# original_sets = [np.where(data == 0, -1, data) for data in original_sets]
original_sets = np.array(original_sets)

testing_sets = [binarize_array(data) for data in test_sets1]
# testing_sets = [np.where(data == 0, -1, data) for data in testing_sets]
testing_sets = np.array(testing_sets)

noisy_sets = [binarize_array(data) for data in test_sets]
noisy_sets = [salt_pepper_noise(data, 0.3) for data in noisy_sets]
# noisy_sets = [np.where(data == 0, -1, data) for data in noisy_sets]
noisy_sets = np.array(noisy_sets)

# network = HopfieldNetwork(784, threshold=50)
# hebbian_training(network, original_sets)
network = HopfieldNetwork(784, threshold=50)
stdp_training(network, original_sets)

testing_predicted = [network.run(data, max_iterations=1000) for data in testing_sets]
noisy_predicted = [network.run(data, max_iterations=1000) for data in noisy_sets]

plot1(original_sets, testing_sets, tag1='Original', tag2='Testing')
plot1(original_sets, noisy_sets, tag1='Original', tag2='Noisy')
plot1(testing_predicted, noisy_predicted, tag1='TestingPredicted', tag2='NoisyPredicted')
# network.plot_weights()

