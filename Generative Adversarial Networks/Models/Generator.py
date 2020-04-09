import tensorflow as tf
Adam = tf.keras.optimizers.Adam
Model = tf.keras.Model
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Conv2d = tf.keras.layers.Conv2D
Input = tf.keras.layers.Input

class Generator:
    def __init__(self, inputs, hiddenDense, outputs):
        self.inputs = inputs
        self.hiddenDense = hiddenDense
        self.outputs = outputs
        self.hiddenActivation = "relu"
        self.outputActivation = "sigmoid"

        self.m = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)

        self.optimizer = Adam(learning_rate=1e-3)

        self.x = Input((self.inputs))
        out = self.createNets(self.x)

        self.out = out * 255

        self.model=Model([self.x], self.out)

        self.vars = self.model.trainable_weights

    def createNets(self, x):
        for hidden in self.hiddenDense:
            x = Dense(hidden, activation=self.hiddenActivation)(x)
        return Dense((28 * 28), activation=self.outputActivation)(x)