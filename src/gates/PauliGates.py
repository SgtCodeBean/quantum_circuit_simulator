import numpy as np
from .Gate import Gate

"""
    Class to define a set of Pauli-gates. Uses generic Gate class to create instances 
    of Pauli-gates.
"""
class PauliGates(Gate):
    X = Gate("X", np.array([[0, 1], [1, 0]]))
    Y = Gate("Y", np.array([[0, -1j], [1j, 0]]))
    Z = Gate("Z", np.array([[1, 0], [0, -1]]))