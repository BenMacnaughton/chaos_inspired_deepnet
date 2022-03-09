import glsneuron
import numpy as np
import multiprocessing as mp




class GLSLayer():
    """
    A class for a GLS layer
    """

    def __init__(self, n, b, q, e):
        """
        Initialize the layer with:
        - n neurons
        - b: the gls map coefficient on the interval [0, 1)
        - q: the initial membrane potential on the interval [0, 1)
        - e: the error range on the interval [0, 1) (typically ≤ 0.1)
        """
        self.n = n
        self.neurons = []
        for _ in range(n):
            self.neurons.append(glsneuron.GLSNeuron(b, q, e))


    def train(self, X, Y):
        """
        Train the layer with a stimulus x
        Returns:
        M - a C x N matrix of the gls maps
        """
        self.classes = list(set(Y))
        self.c = len(self.classes)
        self.M = np.zeros((self.c, self.n))
        self.class_instances = np.zeros(self.c)
        for i, x in enumerate(X):
            c = Y[i]
            self.class_instances[c] += 1
            firing_times = []
            for j, neuron in enumerate(self.neurons):
                firing_times.append(neuron.activate(x[j]))
                neuron.reset()
            self.M[c] += firing_times
        self.M /= self.class_instances[:, None]
        return self.M


    def normalize(self, X):
        """
        Normalize the layer with simuli X
        Returns:
        x_norm - the normalized stimulus
        """
        X_maxs = np.max(X, axis=0)
        X_mins = np.min(X, axis=0)
        X_norm = (X - X_mins) / (X_maxs - X_mins)
        return X_norm


    def predict(self, x):
        """
        Predict the class of a stimulus x
        Returns:
        c - the class of the stimulus
        """
        firing_times = np.zeros(self.n)
        for i, neuron in enumerate(self.neurons):
            firing_times[i] = neuron.activate(x[i])
        # Compute cosine similaries between the stimulus and the gls maps
        cosine_similarities = np.zeros(self.c)
        for c, m in enumerate(self.M):
            norm_yh = np.linalg.norm(firing_times)
            norm_y = np.linalg.norm(m)
            cosine_similarities[c] = np.dot(firing_times, m) / (norm_y * norm_yh)
        # Return the class with the highest cosine similarity
        return self.classes[np.argmax(cosine_similarities)]
  