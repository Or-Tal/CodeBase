# This source code is licensed under MIT license found in the
# LICENSE file in the root directory of this source tree.
# Author: Or Tal.
from solvers.base_solver import BaseSolver


class SolverFactory:

    supported_solvers = {
        "base": BaseSolver
    }

    @staticmethod
    def get_solver(args):
        assert args.solver_name.lower() in SolverFactory.supported_solvers.keys()
        return SolverFactory.supported_solvers[args.solver_name.lower()](args)


