import logging

module_logger = logging.getLogger('H-rep')

from itertools import combinations, chain
from typing import List, Iterable, Tuple, Generator
from ortools.linear_solver import pywraplp
from bidict import bidict
from collections import namedtuple, defaultdict

Var = namedtuple('Var', ('level', 'index'))

Cover = namedtuple('Cover', ('B', 'F'))

class Hrep:
    """Creates the linear H-representation for the given selection problem instance."""

    def __init__(self, input_list: List[int]) -> None:
        self.input = input_list
        self.base_vars = bidict()  # bimaps variable to column position in h-rep
        self.linearisation_vars = bidict()  # bimaps variable pair to column position in h-rep
        for i, length in enumerate(input_list):
            for j in range(length):
                var = Var(level=i, index=j)
                self.base_vars[var] = 1 + len(self.base_vars)

        for i, j in combinations(range(len(input_list)), 2):
            for index_i in range(input_list[i]):
                for index_j in range(input_list[j]):
                    var_i = Var(level=i, index=index_i)
                    var_j = Var(level=j, index=index_j)
                    self.linearisation_vars[(var_i, var_j)] = 1 + len(self.base_vars) + len(self.linearisation_vars)

    def get_glover_woolsey_rep(self) -> str:
        """Returns the h-representation based on the Glover-Woolsey linearisation:
        1) u_ij >= x_i + x_j - 1
        2) u_ij <= x_i
        3) u_ij <= x_j
        """
        num_rows = len(self.input) + 2 * len(self.base_vars) + 4 * len(self.linearisation_vars)
        num_columns = 1 + len(self.base_vars) + len(self.linearisation_vars)
        rep = f'glover-woolsey-linearisation-{self.input}\n'
        rep += 'H-representation\n'
        linearity = str('linearity ') + str(len(self.input)) + ' ' + \
                    ' '.join([str(x) for x in range(1, len(self.input) + 1)])
        rep += linearity + '\n'
        rep += 'begin\n'
        rep += '{} {} integer\n'.format(num_rows, num_columns)
        for card_cons in self._get_all_cardinality_cons(num_columns):
            rep += card_cons
        for nonneg_cons in self._get_all_nonnegativity_cons(num_columns):
            rep += nonneg_cons
        for lowerbound_cons in self._get_all_lowerbound_cons(num_columns):
            rep += lowerbound_cons
        for var1, var2 in self.linearisation_vars.keys():
            cons1, cons2, cons3 = self._get_glover_wolsey_linearisation_cons(num_columns, var1,
            var2)
            rep += self._get_cons_as_string(cons1)
            rep += self._get_cons_as_string(cons2)
            rep += self._get_cons_as_string(cons3)
        rep += 'end'
        return rep

    def get_compact_rep(self) -> str:
        """Returns the h-representation based on the linearisation:
        1) x_i + x_j - y_ij <= 1
        2) 2u_ij <= x_i + x_j
        """
        num_rows = len(self.input) + 2 * len(self.base_vars) + 3 * len(self.linearisation_vars)
        num_columns = 1 + len(self.base_vars) + len(self.linearisation_vars)
        rep = f'compact-linearisation-{self.input}\n'
        rep += 'H-representation\n'
        linearity = str('linearity ') + str(len(self.input)) + ' ' + \
                    ' '.join([str(x) for x in range(1, len(self.input) + 1)])
        rep += linearity + '\n'
        rep += 'begin\n'
        rep += '{} {} integer\n'.format(num_rows, num_columns)
        for card_cons in self._get_all_cardinality_cons(num_columns):
            rep += card_cons
        for nonneg_cons in self._get_all_nonnegativity_cons(num_columns):
            rep += nonneg_cons
        for lowerbound_cons in self._get_all_lowerbound_cons(num_columns):
            rep += lowerbound_cons
        for var1, var2 in self.linearisation_vars.keys():
            cons1, cons2 = self._get_compact_linearisation_cons(num_columns, var1, var2)
            rep += self._get_cons_as_string(cons1)
            rep += self._get_cons_as_string(cons2)
        rep += 'end'
        return rep

    @staticmethod
    def _get_cons_as_string(cons: List[int]) -> str:
        """Returns the given constraint as a str with line break."""
        return ' '.join(map(str, cons)) + '\n'

    def _get_all_cardinality_cons(self, num_columns: int) -> Generator[str, None, None]:
        """Returns cardinality constraints for all sets of the given input."""
        for level, size in enumerate(self.input):
            var_columns = (self.base_vars[Var(level=level, index=i)] for i in range(size))
            cons = self._get_cardinality_cons(num_columns, var_columns, 1)
            yield Hrep._get_cons_as_string(cons)

    @staticmethod
    def _get_cardinality_cons(num_columns: int, columns: Iterable[int], zero_col_val: int) -> List[int]:
        cons = num_columns*[0]
        cons[0] = -zero_col_val
        for i in columns:
            cons[i] = 1
        return cons

    def _get_all_nonnegativity_cons(self, num_columns: int) -> Generator[str, None, None]:
        """Returns non-negativity constraints for all variables."""
        for column in chain(self.base_vars.values(), self.linearisation_vars.values()):
            cons = self._get_nonnegativity_cons(num_columns, column)
            yield Hrep._get_cons_as_string(cons)

    @staticmethod
    def _get_nonnegativity_cons(num_columns: int, var_column: int) -> List[int]:
        """Returns x >= 0 for a variable x."""
        cons = num_columns * [0]
        cons[var_column] = 1
        return cons

    def _get_all_lowerbound_cons(self, num_columns: int) -> Generator[str, None, None]:
        """Returns lowerbound constraints for all base variables."""
        for column in self.base_vars.values():
            cons = self._get_lower_bound_cons(num_columns, column, 1)
            yield Hrep._get_cons_as_string(cons)

    @staticmethod
    def _get_lower_bound_cons(num_columns: int, var_column: int, rhs: int) -> List[int]:
        """Returns x <= rhs for a variable x."""
        cons = num_columns*[0]
        cons[0] = rhs
        cons[var_column] = -1
        return cons

    def _get_glover_wolsey_linearisation_cons(self, num_columns: int, var1: Var, var2: Var) -> Tuple[List[int], List[int], List[int]]:
        """Returns the constraints:
        1) u_ij >= x_i + x_j - 1
        2) u_ij <= x_i
        3) u_ij <= x_j
        """
        position_var1 = self.base_vars[var1]
        position_var2 = self.base_vars[var2]
        position_lin_var = self.linearisation_vars[(var1, var2)]
        cons1 = num_columns*[0]
        cons1[0] = 1
        cons1[position_lin_var] = 1
        cons1[position_var1] = -1
        cons1[position_var2] = -1
        cons2 = num_columns * [0]
        cons2[position_lin_var] = -1
        cons2[position_var1] = 1
        cons3 = num_columns * [0]
        cons3[position_lin_var] = -1
        cons3[position_var2] = 1
        return cons1, cons2, cons3

    def _get_compact_linearisation_cons(self, num_columns: int, var1: Var, var2: Var) -> Tuple[List[int], List[int]]:
        """Returns the constraints:
        1) x_i + x_j - u_ij <= 1
        2) 2u_ij <= x_i + x_j
        """
        position_var1 = self.base_vars[var1]
        position_var2 = self.base_vars[var2]
        position_lin_var = self.linearisation_vars[(var1, var2)]
        cons1 = num_columns * [0]
        cons1[0] = 1
        cons1[position_var1] = -1
        cons1[position_var2] = -1
        cons1[position_lin_var] = 1
        cons2 = num_columns * [0]
        cons2[position_var1] = 1
        cons2[position_var2] = 1
        cons2[position_lin_var] = -2
        return cons1, cons2

    def get_extended_linearisation(self, *, f_weight: float, z_weight: float):
        cover = self._compute_cover_set(f_weight=f_weight, z_weight=z_weight)


    def _compute_cover_set(self, *, z_weight: float, f_weight: float) -> Cover:
        """Create mip model for compute cover set B. See page 5 in 'Compact Linearization for Binary
         Quadratic Problems subject to Assignment Constraints'.
         """
        solver = pywraplp.Solver.CreateSolver('SCIP')
        f_vars = dict()
        z_vars = dict()
        N = self.base_vars.values()
        M = lambda i: (j for j in N if i <= j)
        K = list(range(len(self.input)))
        # create f_ij \in [0,1] for all 1 <= i <= j <= n
        for i in N:
            for j in M(i):
                    f_vars[(i, j)] = solver.NumVar(lb=0., ub=1., name=f'f_{str(i)}{str(j)}')
        # create z_ik \in {0,1} for all k \in K, 1 <= i <= n
        for k in K:
            for i in N:
                z_vars[(i, k)] = solver.BoolVar(name=f'z_{str(i)}{str(k)}')
        # add constraints: f_ij = 1 \forall (i,j) \in E
        for var_i, var_j in self.linearisation_vars.keys():
            i = self.base_vars[var_i]
            j = self.base_vars[var_j]
            solver.Add(f_vars[(i, j)] == 1, name='cons_10')
        # add constraints: f_ij >= z_jk \forall k \in K, i \in A_k, j \in N, i <= j
        for k in K:
            A_k = [index for var, index in self.base_vars.items() if var.level == k]
            for i in A_k:
                for j in M(i):
                    lhs = f_vars[(i, j)]
                    rhs = z_vars[(j, k)]
                    solver.Add(lhs >= rhs, name='cons_11')
        # add constraints: f_ji >= z_jk \forall k \in K, i \in A_k, j \ín N, j < i
        for k in K:
            A_k = [index for var, index in self.base_vars.items() if var.level == k]
            for i in A_k:
                for j in (j for j in N if j < i):
                    lhs = f_vars[(j, i)]
                    rhs = z_vars[(j, k)]
                    solver.Add(lhs >= rhs, name='cons_12')
        # add constraints: \sum_{k: i \in A_k} z_jk >= f_ij \forall 1 <= i <= j <= n
        for i in N:
            for j in M(i):
                k = self.base_vars.inverse[i].level
                lhs = z_vars[(j, k)]
                rhs = f_vars[(i, j)]
                solver.Add(lhs >= rhs, name='cons_13')
        # add constraints: \sum_{k: j \in A_k} z_ik >= f_ij \forall 1 <= i <= j <= n
        for i in N:
            for j in M(i):
                k = self.base_vars.inverse[j].level
                lhs = z_vars[(i, k)]
                rhs = f_vars[(i, j)]
                solver.Add(lhs >= rhs, name='cons_14')

        # add objective
        z_obj = solver.Sum(z_weight*var for var in z_vars.values())
        f_obj = solver.Sum(f_weight*var for var in f_vars.values())
        solver.Minimize(z_obj + f_obj)
        # solve
        status = solver.Solve()
        lp = solver.ExportModelAsLpFormat(False)
        if status == pywraplp.Solver.OPTIMAL:
            B = defaultdict(list)
            for i, k in (key for key, var in z_vars.items() if var.solution_value() > 0.99):
                B[k].append(i)
            F = [key for key, var in f_vars.items() if var.solution_value() > 0.99]
            return Cover(F=F, B=B)
        else:
            raise RuntimeError('Could not optimal solution.')