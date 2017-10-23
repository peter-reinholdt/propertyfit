#!/usr/bin/env python
"""
Contains routines for evaluating potential from
test charges and dipoles, as well as definitions
of cost functions for fitting charges and
isotropic polarizabilities
"""
@jit(nopython=True)
def esp_sum_squared_error(rinvmat, esp_grid_qm, testcharges):
    #compute ESP due to points charges in grid points, get sum of squared error to QM ESP
    natoms      = rinvmat.shape[0]
    ngridpoints = rinvmat.shape[1]
    grid = np.copy(esp_grid_qm)
    for i in range(natoms):
        for j in range(ngridpoints):
            grid[j] -= testcharges[i] * rinvmat[i,j]
    return np.sum(grid**2) / ngridpoints
