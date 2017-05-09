import numpy as np


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


def dipole_potential(dipoles, rinvmat, xyzmat):
    natoms      = rinvmat.shape[0] 
    ngridpoints = rinvmat.shape[1]
    potential = np.zeros(ngridpoints)
    for i in range(natoms):
        for j in range(ngridpoints):
            for k in range(3):
                potential[j] -= dipoles[i,k] * rinvmat[i,j]**3 * xyzmat[i,j,k]
    return potential


def induced_esp_sum_squared_error(rinvmat, xyzmat, induced_esp_grid_qm, field, testpols):
    #induced_esp_grid_qm should be (IND_ESP_QM - ESP_QM_nofield)
    natoms      = rinvmat.shape[0]
    ngridpoints = rinvmat.shape[1]
    grid = np.copy(induced_esp_grid_qm)
    mu_ind = induced_dipole(alpha_ab, field)
    dipole_potential = dipole_potential(mu_ind, rinvmat, xyzmat)
    return np.sum((grid-dipole_potential)**2)

def fit_alpha(structures):
    pass


def fit_alpha_iso(structures):
    #we assume structures contain grid, rinvmat, xyzmat and qm (+induced) ESP
    natoms                  = structures[0].natoms
    alpha_iso = np.zeros((natoms))
    
    def cost(alpha_iso, structures):


