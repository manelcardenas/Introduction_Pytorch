from tkinter.tix import Tree
import torch
import numpy as np


a = torch.ones(5, requires_grad=True)
print(a)
b = a.numpy()
print(type(b))
