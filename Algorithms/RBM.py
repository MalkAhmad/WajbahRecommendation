import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

class RBM(object):

    def __init__(self, visibleDimensions, epochs=20, hiddenDimensions=50, ratingValues=6, learningRate=0.001, batchSize=10):
        self.visibleDimensions = visibleDimensions
        self.epochs = epochs
        self.hiddenDimensions = hiddenDimensions
        self.ratingValues = ratingValues
        self.learningRate = learningRate
        self.batchSize = batchSize

    def Train(self, X):
        ops.reset_default_graph()  # Resets the default graph

        self.MakeGraph()  # Builds the TensorFlow graph

        init = tf.compat.v1.global_variables_initializer()  # Initializes variables
        self.sess = tf.compat.v1.Session()  # Creates a session
        self.sess.run(init)  # Runs the initializer

        np.random.seed(0)  # Set random seed for reproducibility

        for epoch in range(self.epochs):  # Training loop
            np.random.shuffle(X)  # Shuffles the data
            trX = np.array(X)  # Converts to NumPy array
            for i in range(0, trX.shape[0], self.batchSize):  # Batch processing
                self.sess.run(self.update, feed_dict={self.X: trX[i:i+self.batchSize]})  # Runs the update operation
            print("Trained epoch ", epoch)

    def GetRecommendations(self, inputUser):
        hidden = tf.nn.sigmoid(tf.matmul(self.X, self.weights) + self.hiddenBias)  # Computes hidden layer activations
        visible = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.weights)) + self.visibleBias)  # Reconstructs visible layer

        feed = self.sess.run(hidden, feed_dict={self.X: inputUser})  # Gets hidden layer activations
        rec = self.sess.run(visible, feed_dict={hidden: feed})  # Gets reconstructed visible layer
        return rec[0]

    def MakeGraph(self):
        tf.compat.v1.disable_eager_execution()  # Disables eager execution
        tf.random.set_seed(0)  # Sets random seed

        self.X = tf.compat.v1.placeholder(tf.float32, [None, self.visibleDimensions], name="X")  # Placeholder for input

        maxWeight = -4.0 * np.sqrt(6.0 / (self.hiddenDimensions + self.visibleDimensions))  # Weight initialization range
        self.weights = tf.Variable(tf.random.uniform([self.visibleDimensions, self.hiddenDimensions], minval=-maxWeight, maxval=maxWeight), tf.float32, name="weights")  # Weights

        self.hiddenBias = tf.Variable(tf.zeros([self.hiddenDimensions], tf.float32), name="hiddenBias")  # Hidden bias
        self.visibleBias = tf.Variable(tf.zeros([self.visibleDimensions], tf.float32), name="visibleBias")  # Visible bias

        hProb0 = tf.nn.sigmoid(tf.matmul(self.X, self.weights) + self.hiddenBias)  # Hidden layer probabilities
        hSample = tf.nn.relu(tf.sign(hProb0 - tf.random.uniform(tf.shape(hProb0))))  # Hidden layer samples
        forward = tf.matmul(tf.transpose(self.X), hSample)  # Forward pass

        v = tf.matmul(hSample, tf.transpose(self.weights)) + self.visibleBias  # Reconstructs visible layer
        vMask = tf.sign(self.X)  # Mask for observed values
        vMask3D = tf.reshape(vMask, [tf.shape(v)[0], -1, self.ratingValues])  # Reshapes mask
        vMask3D = tf.reduce_max(vMask3D, axis=[2], keepdims=True)  # Reduces along rating dimension
        v = tf.reshape(v, [tf.shape(v)[0], -1, self.ratingValues])  # Reshapes visible layer
        vProb = tf.nn.softmax(v * vMask3D)  # Softmax activation
        vProb = tf.reshape(vProb, [tf.shape(v)[0], -1])  # Reshapes back
        hProb1 = tf.nn.sigmoid(tf.matmul(vProb, self.weights) + self.hiddenBias)  # Hidden layer probabilities (reconstructed)
        backward = tf.matmul(tf.transpose(vProb), hProb1)  # Backward pass

        weightUpdate = self.weights.assign_add(self.learningRate * (forward - backward))  # Weight update
        hiddenBiasUpdate = self.hiddenBias.assign_add(self.learningRate * tf.reduce_mean(hProb0 - hProb1, 0))  # Hidden bias update
        visibleBiasUpdate = self.visibleBias.assign_add(self.learningRate * tf.reduce_mean(self.X - vProb, 0))  # Visible bias update

        self.update = [weightUpdate, hiddenBiasUpdate, visibleBiasUpdate]  # Update operations
