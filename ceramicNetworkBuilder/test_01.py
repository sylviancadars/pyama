import numpy as np
from scipy.optimize import minimize

nbrs = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

def fun(v):
    (x, y, z) = v
    v = np.asarray(v)
    result = 0.0
    for nbr in nbrs:
        result += 1/np.linalg.norm(v-nbr)
    return result

cons = ({'type': 'eq', 'fun': lambda v: np.linalg.norm(v) - 1})

result = minimize(fun, [0, 0, 0], constraints=cons)
if result.success:
    print('x = {}'.format(result.x))
else:
    print('minimimization failed.')

