import numpy as np

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F


class GLSLayer(nn.Module):

    def __init__(self, b: torch.Tensor, q: torch.Tensor, e: torch.Tensor, classes: int, neurons: int):
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
        self.M = nn.Parameter(torch.zeros((classes, neurons)))
        self.M.requires_grad = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def fire_neurons(self, x: torch.Tensor):
        m = torch.ones(x.shape, device=self.device) * self.q # m.shape is len(x) x len(x[0])
        k = torch.ones(m.shape, device=self.device)
        
        for idx in range(len(m)):
            m_indices = torch.nonzero(m[idx])
            last_indices = m_indices.cpu()

            counter = 0
            
            while len(m_indices):
                m[idx] = torch.where(
                    m[idx] < self.b, # maybe all need to be less than self.b?
                    m[idx] / self.b,
                    (1 - m[idx]) / (1 - self.b)
                ) # Condition, True, False
                under = m[idx] < x[idx] - self.e 
                over = m[idx] > x[idx] + self.e
                invalid = under + over
                m_indices = np.intersect1d(torch.nonzero(invalid).cpu(), last_indices)
                last_indices = m_indices
                k[idx, m_indices] += 1

            print("Sample {} Converged".format(idx + 1))

        return k


    def forward(self, x: torch.Tensor, return_logits: bool = False):
        k = self.fire_neurons(x)

        classes = []
        logits = torch.zeros(x.shape[0], self.M.shape[0], device=self.device)
        
        for idx, sample in enumerate(k):
            similarities = F.cosine_similarity(sample, self.M)
            logits[idx] = similarities
            classes.append(torch.argmax(similarities).cpu())

        if return_logits:
            return logits

        return torch.tensor(classes)


    def train(self, x: torch.Tensor, y: torch.Tensor):
        k = self.fire_neurons(x)
        occurences = torch.zeros(self.M.shape[0], device=self.device)

        for idx in range(len(k)):
            self.M[y[idx]] += k[idx]
            occurences[y[idx]] += 1

        self.M /= occurences[:, None]


class Trainer():

    def __init__(self, model: GLSLayer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)


    def normalize(self, x: np.array):
        min_x = np.min(x)
        x_ = x - min_x
        return x_ / (np.max(x) - min_x)


    def load_data(self, x: np.array, y: np.array, test_size=0.2, random_state=None):
        x = self.normalize(x)
        p = np.random.permutation(len(x))
        X = x[p]
        Y = y[p]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        self.X_train = torch.tensor(X_train).float().to(self.device)
        self.X_test = torch.tensor(X_test).float().to(self.device)
        self.y_train = torch.tensor(y_train).to(self.device)
        self.y_test = torch.tensor(y_test).to(self.device)


    def train(self):
        self.model.train(self.X_train, self.y_train)
