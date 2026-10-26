import numpy as np
'''
Solve the linear system
'''
def solve_linear_system(K, F):
    """
    solve the linear system K * u = F
    :param K: global stiffness matrix
    :param F: load vector
    :return: solution vector u
    """
    u = np.linalg.solve(K, F)
    return u