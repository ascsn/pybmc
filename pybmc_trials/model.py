import numpy as np

class Model:
    """
    A class representing a model with a domain (x) and an output (y).

    Attributes:
        name (str): The name of the model.
        x (np.ndarray): The domain of the model.
        y (np.ndarray): The output of the model.
    """

    def __init__(self, name, x, y):
        """
        Initialize the Model object.

        Args:
            name (str): The name of the model.
            x (array-like): The domain of the model.
            y (array-like): The output of the model.
        """
        self.name = name
        self.x = np.array(x)
        self.y = np.array(y)
