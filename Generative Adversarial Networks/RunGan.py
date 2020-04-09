#imports
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import random
import copy
from datetime import datetime

from Models.GenerativeAdversarialNet import GAN


#dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(*x_train.shape, 1)


new_lables_dict = {0: -1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: -1, 7: -1, 8: -1, 9: -1}
labels = copy.deepcopy(y_train)
for i, label in enumerate(y_train):
    labels[i] = new_lables_dict[label]

new_x = []
for data, label in zip(x_train, y_train):
    if label == 1:
        new_x.append(data)

x_train = new_x


#variables
epochs = 1000
iters = 20
batch_size = 64
hiddenDenseLayers = [200, 500]
hiddenCNNLayers = [(64, 3)]
input_size = x_train[0].shape

if_20_test_every = 1
if_100_test_every = 5
if_1000_test_every = 20

#create network
gan = GAN(input_size, hiddenCNNLayers, hiddenDenseLayers)

Discriminater_model = gan.Discriminator
Generator_model = gan.Generator

#mainloop
data_amount = iters*batch_size

def avg(x):
    total = 0
    for char in x:
        total += x
    return total / len(x)

for i in range(epochs):
    #create data, use a mix of real data and the created data
    #real data
    np.random.shuffle(x_train)
    real_data = x_train[:data_amount]

    #fake data
    noise = np.random.rand(data_amount, 100)
    fake_data = gan.Generator.model.predict(noise).reshape(data_amount, *input_size)

    #total data
    total_data = np.array((*real_data, *fake_data))

    #train the Discriminator
    labels = np.array((*(np.ones(data_amount)*10), *(-np.ones(data_amount)*10)))
    Discriminater_model.train(total_data, labels)

    #train the Generator
    #loss: D(x) - D(G(z)) x being real input and z being noise
    gan.set_weigths()
    loss = Discriminater_model.model.predict(total_data)
    outputs = (loss[:data_amount])
    genLoss = gan.train_gan(noise, outputs)

    if (i < 20 and i % if_20_test_every == 0) or (i > 20 and i < 100 and i % if_100_test_every == 0) or (i > 100 and i % if_1000_test_every == 0):
        plt.imshow(fake_data[10].reshape(28, 28), cmap="gist_gray")
        plt.savefig('created_1s/numberat%ssteps.png' % (i))

    print("training steps: %s, average normal data: %.2f, average generated data: %.2f, generator loss: %.2f" % (i, avg(loss[:data_amount][0]), avg(loss[data_amount:][0]), genLoss))