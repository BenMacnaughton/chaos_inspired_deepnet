

class GLSNeuron():
    """
    A class for a single neuron in a GLS layer.
    """

    def __init__(self, b, q, e):
        """
        Itiialize the neuron with:
        - b: the gls map coefficient on the interval [0, 1)
        - q: the initial membrane potential on the interval [0, 1)
        - m: the current membrane potential on the interval [0, 1) (initiallly q)
        - e: the error range on the interval [0, 1) (typically ≤ 0.1)
        - k: the number of times the neuron has fired (initially 0)
        """
        self.b = b
        self.q = q
        self.m = q
        self.e = e
        self.k = 0


    def fire(self):
        """
        Fire the neuron.
        """
        if self.m < self.b:
            self.m /= self.b
        else:
            self.m = (1 - self.m) / (1 - self.b)
        self.k += 1


    def activate(self, x):
        """
        Activate the neuron with a stimulus x
        The neuron fires until its membrane potential
        in the neighborhood of (x - e, x + e)
        Returns:
        k - the number of times the neuron has fired
        """
        while(self.m < x - self.e or self.m > x + self.e):
            self.fire()
        return self.k


    def reset(self):
        """
        Reset the neuron.
        """
        self.m = self.q
        self.k = 0
