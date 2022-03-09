import glsneuron

class GLSLayer():
    """
    A class for a GLS layer
    """

    def __init__(self, n, b, q, e):
        """
        Initialize the layer with n neurons
        """
        self.n = n
        self.neurons = []
        for _ in range(n):
            self.neurons.append(glsneuron.GLSNeuron(b, q, e))
