structures = chargefit.loadfchks("test")
[s.compute_grid() for s in structures]
[s.compute_rinvmat() for s in structures]
[s.compute_qm_esp() for s in structures]

q0 = np.zeros(s[0].natoms)
def cost(q):
    return np.average([esp_sum_squared_error(s.rinvmat, s.esp_grid_qm, q) for s in structures])


