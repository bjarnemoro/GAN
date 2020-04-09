import tensorflow as tf
Model = tf.keras.Model

from Models.Discriminator import Discriminator
from Models.Generator import Generator

class GAN:
    def __init__(self, inputs, hiddenCNN, hiddenDense):
        self.Generator = Generator(100, hiddenDense, inputs)
        self.Discriminator = Discriminator(inputs, hiddenCNN, hiddenDense)

        self.disc_layers = (1, 3, 4, 5)
        self.complete_layers = (9, 11, 12, 13)

        genOut = tf.reshape(self.Generator.out, [tf.shape(self.Generator.out)[0], 28, 28, 1])
        self.completeModel = Model([self.Generator.x], self.Discriminator.createNets(genOut))

        self.loss = lambda: tf.reduce_mean((self.completeModel(self.noise) - self.outputs)**2)

    def train_disc(self, data, labels):
        self.Discriminator.train(data, labels)

    def train_gan(self, noise, outputs):
        self.noise = noise
        self.outputs = outputs
        with tf.GradientTape() as tape:
            loss = self.loss()
            vars = self.Generator.vars
            grads = tape.gradient(loss, vars)
            grads_and_vars = zip(grads, vars)
            self.Generator.optimizer.apply_gradients(grads_and_vars)

        return loss

    def set_weigths(self):
        for disc, complete in zip(self.disc_layers, self.complete_layers):
            self.completeModel.layers[complete].set_weights(self.Discriminator.model.layers[disc].get_weights())