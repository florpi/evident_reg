import inspect
from typing import Callable
import jax.numpy as np
from jax import jit, grad


def maximize(
    func: Callable,
    weights: np.array,
    argnums: int = -1,
    learning_rate: float = 0.01,
    n_epochs: int = 10,
    **kwargs,
)->np.array:
    """ Maximize a function with respect to a set of weights

    Args:
        func (Callable): function to maximize 
        weights (np.array): initial weights 
        argnums (int, optional): where the weights argument appears in func (always last). Defaults to -1.
        learning_rate (float, optional): learning rate value. Defaults to 0.01.
        n_epochs (int, optional): number of epochs to run it for. Defaults to 10.

    Returns:
        np.array: optimal weights that maximize func 
    """
    # jit gradients function
    ordered_kwargs = [
        kwargs[key]
        for key in inspect.signature(func).parameters.keys()
        if key != "weights"
    ]
    # func = jit(func, static_argnums=(0,))
    grad_X = grad(func, argnums=argnums)
    grad_X = jit(grad_X)
    for i in range(n_epochs):
        weights += learning_rate * grad_X(*ordered_kwargs, weights)
        if i % 2 == 0:
            print(func(**kwargs, weights=weights))
    return weights
