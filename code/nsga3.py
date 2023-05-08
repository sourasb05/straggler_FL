from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.factory import get_problem
from pymoo.problems.functional import FunctionalProblem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np

objs = [
    lambda x: np.subtract(tm,mr)
    lambda x: np.divide(np.multiply(td, 8192) , np.multiply(rwt*3600))
    lambda x:
    ]


problem = my_problem()

pf = problem.pareto_front()
ps = problem.pareto_set()

ref_points = np.array([[0.5, 0.2], [0.1, 0.6]])
algorithm = RNSGA2(
    ref_points=ref_points,
    pop_size=40,
    epsilon=0.01,
    normalization='front',
    extreme_points_as_reference_points=False,
    weights=np.array([0.5, 0.5]))

res = minimize(problem,
               algorithm,
               save_history=True,
               termination=('n_gen', 250),
               seed=1,
               pf=pf,
               disp=False,
               verbose=True)
print("pareto front:",pf)
print("pareto set:",ps)
Scatter().add(pf, label="pf").add(res.F, label="F").add(ref_points, label="ref_points").show()
# plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, color="red")
# plot.show()
