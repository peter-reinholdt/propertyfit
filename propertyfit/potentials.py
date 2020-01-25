import numpy as np
from math import factorial
from string import ascii_lowercase as abc
from string import ascii_uppercase as ABC
from numpy import sqrt
from .utilities import memoize_on_first_arg_function

def T0(Ra, Rb):
    Rab = -(Ra[:, np.newaxis, :] - Rb[np.newaxis, :, :])
    x = Rab[..., 0]
    y = Rab[..., 1]
    z = Rab[..., 2]
    shape = Rab.shape[:-1] + (3, ) * 0
    result = np.zeros(shape, dtype=np.float64)
    result[..., ] = 1 / sqrt(x**2 + y**2 + z**2)
    return result

def T1(Ra, Rb):
    Rab = -(Ra[:, np.newaxis, :] - Rb[np.newaxis, :, :])
    x = Rab[..., 0]
    y = Rab[..., 1]
    z = Rab[..., 2]
    shape = Rab.shape[:-1] + (3, ) * 1
    result = np.zeros(shape, dtype=np.float64)
    x0 = (x**2 + y**2 + z**2)**(-3 / 2)
    result[..., 0] = -x * x0
    result[..., 1] = -x0 * y
    result[..., 2] = -x0 * z
    return result

def T2(Ra, Rb):
    Rab = -(Ra[:, np.newaxis, :] - Rb[np.newaxis, :, :])
    x = Rab[..., 0]
    y = Rab[..., 1]
    z = Rab[..., 2]
    shape = Rab.shape[:-1] + (3, ) * 2
    result = np.zeros(shape, dtype=np.float64)
    x0 = x**2
    x1 = y**2
    x2 = z**2
    x3 = x1 + x2
    x4 = (x0 + x3)**(-5 / 2)
    x5 = 3 * x * x4
    x6 = x5 * y
    x7 = x5 * z
    x8 = 3 * x4 * y * z
    result[..., 0, 0] = -x4 * (-2 * x0 + x3)
    result[..., 0, 1] = x6
    result[..., 0, 2] = x7
    result[..., 1, 0] = x6
    result[..., 1, 1] = -x4 * (x0 - 2 * x1 + x2)
    result[..., 1, 2] = x8
    result[..., 2, 0] = x7
    result[..., 2, 1] = x8
    result[..., 2, 2] = -x4 * (x0 + x1 - 2 * x2)
    return result

def field(structure, multipole_rank, multipoles, field_rank, mask=None):
    """
    Calculates the field (derivatives) due to a set of multipoles in a set of points
    
    idx: index used to cache Tn tensors
    Rab: distance ndarrray
        shape is (nmult,...,3) so we can work with both a single distance vector (nmult,3), a vector of distance vectors (nmult,N,3) and so on
    multipole_rank: rank of the multipole
    multipoles: ndarray, cartesian multipole moments of the same rank, shape (number_of_multipoles) + (3)*multipole_rank
        0-> (number_of_multipoles,)    charge
        1-> (number_of_multipoles,3)   dipole
        2-> (number_of_multipoles,3,3) quadrupole
        ...
    field_rank: rank of output field
        0->  potential (negative of, depending on definitions)
        1->  field
        2->  field gradient
        ...
    """
    tensor_rank = field_rank + multipole_rank
    multipole_additional_rank = len(multipoles.shape) - multipole_rank
    Rab_additional_rank = 3 - multipole_additional_rank - 1
    Tn = getattr(structure, f'T{tensor_rank}')
    factor = -1.0 * (-1)**multipole_rank / factorial(multipole_rank)
    # apply mask
    if mask is not None:
        Tn = Tn[mask, ...]
        multipoles = multipoles[mask, ...]
    # Tn:multipoles->out
    # abc... for multipole/field contraction
    # ABC... for additional ranks
    A = ABC[0:multipole_additional_rank + Rab_additional_rank] + abc[:tensor_rank]
    B = ABC[0:multipole_additional_rank] + abc[tensor_rank - multipole_rank:tensor_rank]
    C = ABC[multipole_additional_rank:multipole_additional_rank + Rab_additional_rank] + abc[:field_rank]
    signature = "{},{}->{}".format(A, B, C)
    res = factor * np.einsum(signature, Tn, multipoles)
    return res
