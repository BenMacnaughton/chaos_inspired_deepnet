import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                            confusion_matrix,
                            f1_score)
import numpy as np


class Trainer():

    def __init__(self, model: torch.nn.Module):
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
        self.y_test = torch.tensor(y_test)


    def train(self):
        self.model.train(self.X_train, self.y_train)


    def evaluate(self):
        predictions = self.model(self.X_test)
        acc = accuracy_score(self.y_test, predictions)
        conf = confusion_matrix(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average='macro')

        return acc, conf, f1
