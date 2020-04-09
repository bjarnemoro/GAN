import tensorflow as tf
Adam = tf.keras.optimizers.Adam
Model = tf.keras.Model
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Conv2d = tf.keras.layers.Conv2D
Input = tf.keras.layers.Input


class Discriminator:
    def __init__(self, inputs, hiddenCNN, hiddenDense):
        self.inputs = inputs
        self.hiddenCNN = hiddenCNN
        self.hiddenDense = hiddenDense
        self.hiddenActivation = "relu"
        self.outputActivation = None

        self.m = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)

        self.optimizer = Adam(learning_rate=1e-3)

        x = Input((self.inputs))
        out = self.createNets(x)

        self.model=Model(x, out)

        self.loss = lambda: tf.reduce_mean((self.model(self.inputs)-self.labels)**2)
        self.vars = self.model.trainable_weights

    def train(self, data, labels):
        self.inputs = self.m(data)
        self.labels = self.m(labels.reshape(*labels.shape, 1))

        #print(tf.multiply(self.model.predict(self.inputs), self.labels))

        with tf.GradientTape() as tape:
            loss = self.loss()
        grads = tape.gradient(loss, self.vars)
        grads_and_vars = zip(grads, self.vars)
        self.optimizer.apply_gradients(grads_and_vars)

    def createNets(self, x):
        for filters, kernels in self.hiddenCNN:
            x = Conv2d(filters=filters, kernel_size=kernels, activation=self.hiddenActivation)(x)
        if type(self.inputs) is not int:
            x = Flatten()(x)
        for hidden in self.hiddenDense:
            x = Dense(hidden, activation=self.hiddenActivation)(x)
        return Dense(1, activation=self.outputActivation)(x)