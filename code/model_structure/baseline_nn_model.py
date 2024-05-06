import random
import torch
import math
class MLP_t(torch.nn.Module):
    def __init__(self, num_class):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()
        self.l1 = torch.nn.Linear(384, num_class)
        # self.l2 = torch.nn.Linear(20, num_class)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 4, 5
        and reuse the e parameter to compute the contribution of these orders.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same parameter many
        times when defining a computational graph.
        """
        y = self.relu(self.l1(x))
        # y = self.l2(y)

        return y


class MLP_Combined(torch.nn.Module):
    def __init__(self, num_class):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()
        self.l1 = torch.nn.Linear(3456, 1000)
        # self.l2 = torch.nn.Linear(384, 20)
        # self.l3 = torch.nn.Linear(384, 20)
        # self.l4 = torch.nn.Linear(384, 20)
        # self.l5 = torch.nn.Linear(384, 20)
        # self.l6 = torch.nn.Linear(384, 20)
        # self.l7 = torch.nn.Linear(384, 20)
        # self.l8 = torch.nn.Linear(384, 20)
        # self.l9 = torch.nn.Linear(384, 20)

        self.l2 = torch.nn.Linear(1000, 1000)
        self.l3 = torch.nn.Linear(1000, 1000)
        self.l4 = torch.nn.Linear(1000, num_class)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 4, 5
        and reuse the e parameter to compute the contribution of these orders.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same parameter many
        times when defining a computational graph.
        """
        y = self.relu(self.l1(x))
        y = self.relu(self.l2(y))
        y = self.relu(self.l3(y))
        y = self.relu(self.l4(y))
        # y = self.l2(y)

        return y
    
#%%


class MLP(torch.nn.Module):
    def __init__(self, num_class):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()
        self.l1 = torch.nn.Linear(41512, num_class)
        # self.l2 = torch.nn.Linear(384, 20)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 4, 5
        and reuse the e parameter to compute the contribution of these orders.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same parameter many
        times when defining a computational graph.
        """
        y = self.relu(self.l1(x))

        return y