import numpy as np


def bisector(point1, point2, point3):
    """
    decide local frame unit vectors x, y, z
    point1: coordinates of the atom in the origin of the local coordinate system
    point2: coordinates of first atom to define bisector
    point3: coordinates of second atom to define bisector
    """
    #create z-axis
    v1 = point2 - point1
    v1 = v1 / np.linalg.norm(v1)
    v2 = point3 - point1
    v2 = v2 / np.linalg.norm(v2)
    z = v1 + v2
    z = z / np.linalg.norm(z)

    #create x-axis by rejection
    x = point3 - point1
    x = x - (np.dot(x, z) / np.dot(z, z)) * z
    x = x / np.sqrt(np.sum(x * x))

    #dot = np.dot(v2, z)
    #x = v2 - dot * z
    #x = x / np.linalg.norm(x)

    #right hand rule for y
    y = np.cross(z, x)
    return np.vstack([x, y, z]).T


def zthenx(point1, point2, point3):
    """
    decide local frame unit vectors x, y, z
    point1: coordinates of the atom in the origin of the local coordinate system
    point2: coordinates of the atom to which the local z-axis is created
    point3: coordinates of a third atom, with which the local x-axis is created 
    """

    #create z-axis
    z = point2 - point1
    z = z / np.sqrt(np.sum(z * z))

    #create x-axis by rejection
    x = point3 - point1
    x = x - (np.dot(x, z) / np.dot(z, z)) * z
    x = x / np.sqrt(np.sum(x * x))

    #right hand rule for y
    y = np.cross(z, x)

    #rotation matrix
    return np.vstack([x, y, z]).T
