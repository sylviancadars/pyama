import sympy
from time import perf_counter

def print_solutions(solutionsi, calc_time=None):
    if calc_time is not None:
        print('{} solution(s) found in {:.3f} s.'.format(len(solutions),
                                                         calc_time))
    else:
        print('{} solution(s) found.'.format(len(solutions)))
    print('solutions = {} of type {}'.format(solutions, type(solutions)))
    for solution in solutions:
        print('solution = {} of type {}'.format(solution, type(solution)))
        print('type(solution[0]) = ', type(solution[0]))


def equations(p):
    x, y = p
    eq1 = x**2 + (y-4)**2 - 1.0
    eq2 = (x-5)**2 + y**2 + 9.0
    return (eq1, eq2)

# (x, y) = sympy.symbols('x, y', real=True)
(x, y) = sympy.symbols('x, y', real=False)

print('\n' + 50*'-' + '\nTest with sympy.nonlinsolve()\n' + 50*'-')
tic = perf_counter()
solutions = sympy.nonlinsolve(equations((x, y)), (x, y))
tac = perf_counter()
print_solutions(solutions, calc_time=tac-tic)

print('\n' + 50*'-' + '\nTest with sympy.solve()\n' + 50*'-')
tic = perf_counter()
solutions = sympy.solve(equations((x, y)), (x, y))
tac = perf_counter()
print_solutions(solutions, calc_time=tac-tic)


from scipy.optimize import fsolve

print('\n' + 50*'-' + '\nTest with scipy.fsolve()\n' + 50*'-')
tic = perf_counter()
x, y =  fsolve(equations, (0, 0))
print('Solution found in {} s.'.format(perf_counter()-tic))
print(x, y)



