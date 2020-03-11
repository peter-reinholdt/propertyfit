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


def triple(A, B, C):
    return np.dot(A, np.cross(B, C))


def normvec(A, B):
    AB = A - B
    return AB / np.linalg.norm(AB)


def internal_four_neighbors(A, B1, B2, B3, B4):
    # example: aliphatic carbon atom, quarternary nitrogen
    z = normvec(B1, A) + normvec(B2, A) + normvec(A, B3) + normvec(A, B4)
    z = z / np.linalg.norm(z)
    y = np.cross(z, np.cross(normvec(A, B3) + normvec(B4, A), z))
    y = y / np.linalg.norm(y)
    if triple(normvec(B4, B1), normvec(B4, B2), normvec(B4, B3)):
        x = np.cross(y, z)
    else:
        x = -np.cross(y, z)
    return np.vstack([x, y, z]).T


def internal_four_neighbors_symmetric(A, B1a, B1b, B1c, B4):
    # example: methyl carbon
    z = normvec(A, B4) + 3 * (normvec(B1a, A) + normvec(B1b, A) + normvec(B1c, A))
    z = z / np.linalg.norm(z)
    y = np.cross(z, np.cross(normvec(A, B1a), z))
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    return np.vstack([x, y, z]).T


def internal_three_neighbors(A, B1, B2, B3):
    # example: aromatic carbon, N with three neighbors
    z = np.cross(normvec(B3, B1), normvec(B3, B2))
    z = z / np.linalg.norm(z)
    if np.dot(normvec(B1, A) + normvec(B2, A) + normvec(B3, A), z) > 0:
        z = -z
    y = np.cross(z, np.cross(normvec(B1, A) + normvec(B2, A) + normvec(A, B3), z))
    y = y / np.linalg.norm(y)
    if np.dot(np.cross(y, z), normvec(B1, A)) > 0:
        x = np.cross(y, z)
    else:
        x = -np.cross(y, z)
    return np.vstack([x, y, z]).T


def internal_two_neighbors(A, B1, B2):
    # example: ether oxygen, hydroxyl oxygen
    z = np.cross(normvec(A, B1), normvec(A, B2))
    z = z / np.linalg.norm(z)
    y = np.cross(z, np.cross(normvec(B1, A) + normvec(B2, A), z))
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    return np.vstack([x, y, z]).T


def terminal_three_adjacent_neighbors(A, B1, B2, B3, B4):
    # example: hydrogen, halogen on aliphatic carbon
    z = normvec(B1, A)
    y = np.cross(z, np.cross((normvec(B1, B4) + normvec(B2, B1) + normvec(B3, B1)), z))
    y = y / np.linalg.norm(y)
    if triple(normvec(B1, B2), normvec(B1, B3), normvec(B1, B4)):
        x = np.cross(y, z)
    else:
        x = -np.cross(y, z)
    return np.vstack([x, y, z]).T


def terminal_two_adjacent_neighbors(A, B1, B2, B3):
    # example: aromatic hydrogen/halogen, carbonyl oxygen, amide hydrogen
    z = normvec(B1, A)
    y = np.cross(z, np.cross(np.cross(normvec(B2, A), normvec(B3, A)), z))
    y = y / np.linalg.norm(y)
    if np.dot(
            normvec(A, B1) + normvec(B2, B1) + normvec(B3, B1),
            np.cross(z, np.cross(np.cross(normvec(B2, A), normvec(B3, A)), z))) > 0:
        y *= -1
    if np.dot(np.cross(y, z), normvec(A, B2)) > 0:
        x = np.cross(y, z)
    else:
        x = -np.cross(y, z)
    return np.vstack([x, y, z]).T


def terminal_one_adjacent_neighbor(A, B1, B2):
    # example: hydroxyl H
    z = normvec(B1, A)
    y = np.cross(z, normvec(B2, B1))
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    return np.vstack([x, y, z]).T
