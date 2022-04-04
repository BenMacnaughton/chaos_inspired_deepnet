import numpy as np

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F


class GLSLayer(nn.Module):

    def __init__(self, b: torch.Tensor, q: torch.Tensor, e: torch.Tensor, classes: int):
        """Create an instance of a GLSLayer.

        Args:
            b (torch.Tensor): topological transitivity coefficient (size: (len(neurons), ))
            q (torch.Tensor): Membrane potential (size: (len(neurons), ))
            e (torch.Tensor): epsilon of neighborhood (size: (len(neurons), ))
            classes (int): number of classes in output
        """
        super(GLSLayer, self).__init__()
        self.b = nn.Parameter(b)
        self.q = nn.Parameter(q)
        self.e = nn.Parameter(e)
        self.M = nn.Parameter(torch.zeros((classes, b.shape[0])))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def fire_neurons(self, x: torch.Tensor):
        m = torch.ones(x.shape[0], 1, device=self.device) * self.q # m.shape is len(x) x len(q)
        k = torch.ones(m.shape, device=self.device)
        m_indices = torch.nonzero(m)

        while len(m_indices):
            m = torch.where(m < self.b, m / self.b, (1 - m) / (1 - self.b)) # Condition, True, False
            under = m < x - self.e 
            over = m > x + self.e
            invalid = under + over
            m_indices = torch.nonzero(invalid)
            k[m_indices[:, 0], m_indices[:, 1]] += 1

        return k


    def forward(self, x: torch.Tensor, return_logits: bool = False):
        k = self.fire_neurons(x)

        classes = []
        logits = torch.zeros(x.shape[0], self.M.shape[0], device=self.device)
        
        for idx, sample in enumerate(k):
            similarities = F.cosine_similarity(sample, self.M)
            logits[idx] = similarities
            classes.append(torch.argmax(similarities))

        if return_logits:
            return logits

        return torch.tensor(classes)


    def train(self, x: torch.Tensor, y: torch.Tensor):
        k = self.fire_neurons(x)
        occurences = torch.zeros(self.M.shape[0])

        for idx in range(len(k)):
            self.M[y[idx]] += k[idx]
            occurences[y[idx]] += 1

        self.M /= occurences[:, None]


class Trainer():

    def __init__(self, model: GLSLayer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)


    def load_data(self, x: np.array, y: np.array, test_size=0.2, random_state=None):
        p = np.random.permutation(len(x))
        X = x[p]
        Y = y[p]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        self.X_train = torch.tensor(X_train).float().to(self.device)
        self.X_test = torch.tensor(X_test).float().to(self.device)
        self.y_train = torch.tensor(y_train).float().to(self.device)
        self.y_test = torch.tensor(y_test).float().to(self.device)


    def train(self):
        self.model.train(self.X_train, self.y_train)
