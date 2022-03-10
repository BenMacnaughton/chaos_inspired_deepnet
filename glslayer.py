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


    def predict_all(self, X):
        """
        Predict the class of all stimuli X
        Returns:
        Y - the classes of the stimuli
        """
        Y = []
        for x in X:
            Y.append(self.predict(x))
        return Y


    def train_test_split(self, X, Y, m):
        """
        Shuffle datasets and get
        a training set with m stimuli and a test set with the rest
        Returns:
        X_train, Y_train, X_test, Y_test
        """
        p = np.random.permutation(len(X))
        X = X[p]
        Y = Y[p]

        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        for i, x in enumerate(X):
            if i % m == 0:
                X_test.append(x)
                Y_test.append(Y[i])
            else:
                X_train.append(x)
                Y_train.append(Y[i])
        return X_train, Y_train, X_test, Y_test


    def train_test(self, X, Y, m):
        """
        Train the layer with a training set and test it with a test set
        Returns:
        M_train, M_test - the training and test gls maps
        """
        X_norm = self.normalize(X)
        X_train, Y_train, X_test, Y_test = self.train_test_split(X_norm, Y, m)
        self.train(X_train, Y_train)
        acc = 0
        for i, x in enumerate(X_test):
            if self.predict(x) == Y_test[i]:
                acc += 1
        return self.M, acc / len(X_test)
