import numpy as np
from numba import jit

def induced_dipole(alpha_ab, field):
    """Anisotropic atom-centered dipole-dipole polarizabilities in a homogenous field"""
    natoms = len(alpha_ab)
    mu = np.zeros((natoms,3))
    for i in range(natoms):
        for j in range(3):
            for k in range(3):
                #mu_i,alpha = alpha_i,alphabeta*Fbeta
                mu[i,j] += alpha_ab[i,j,k] * field[k]
    return mu

@jit #numba makes this hella fast
def dipole_potential(dipoles, rinvmat, xyzmat):
    natoms      = rinvmat.shape[0] 
    ngridpoints = rinvmat.shape[1]
    potential = np.zeros(ngridpoints)
    for i in range(natoms):
        for j in range(ngridpoints):
            for k in range(3):
                potential[j] -= dipoles[i,k] * rinvmat[i,j]**3 * xyzmat[i,j,k]
    return potential


def induced_esp_sum_squared_error(rinvmat, xyzmat, induced_esp_grid_qm, field, alpha_ab):
    #induced_esp_grid_qm should be (IND_ESP_QM - ESP_QM_nofield)
    natoms      = rinvmat.shape[0]
    ngridpoints = rinvmat.shape[1]
    grid = np.copy(induced_esp_grid_qm)
    mu_ind = induced_dipole(alpha_ab, field)
    alpha_pot = dipole_potential(mu_ind, rinvmat, xyzmat)
    return np.sum((grid-alpha_pot)**2)


def cost_alpha_iso(refstructures, fieldstructures, alpha_iso):
    #we assume structures contain grid, rinvmat, xyzmat and qm ESP
    #refstructures, fieldstructures arranged so they line up pairwise, eg:
    #VAL_0   VAL_0 x+50
    #VAL_0   VAL_0 z-60
    #...
    #VAL_1   VAL_1 z+10
    #alpha_iso is of size natoms
    #returns RMSD
    natoms = refstructures[0].natoms
    
    #reality check
    assert(len(refstructures) == len(fieldstructures))
    assert(natoms == len(alpha_iso))

    #fill in isotropic polarizabilities in full tensor
    alphas = np.zeros((natoms,3,3))
    for i in range(natoms):
        alphas[i,0,0] = alpha_iso[i]
        alphas[i,1,1] = alpha_iso[i]
        alphas[i,2,2] = alpha_iso[i]

    cost = 0.0
    npoints = 0
    for i in range(len(refstructures)):
        rs = refstructures[i] 
        fs = fieldstructures[i]
        cost += induced_esp_sum_squared_error(rs.rinvmat,  
                                              rs.xyzmat, 
                                              rs.esp_grid_qm - fs.esp_grid_qm,
                                              fs.field,
                                              alphas)
        npoints += len(rs.grid)

    return np.sqrt(cost/npoints)
