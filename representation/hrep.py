import logging

module_logger = logging.getLogger('H-rep')

from itertools import combinations, chain
from typing import List, Iterable, Tuple, Generator

from collections import namedtuple

Var = namedtuple('Var', ('set', 'index'))

class Hrep:
    """Creates the linear H-representation for the given selection problem instance."""

    def __init__(self, input_list: List[int]) -> None:
        self.input = input_list
        self.base_vars = dict()  # maps variable to column position in h-rep
        self.linearisation_vars = dict()  # maps variable pair to column position in h-rep
        for i, length in enumerate(input_list):
            for j in range(length):
                var = Var(set=i, index=j)
                self.base_vars[var] = 1 + len(self.base_vars)

        for i, j in combinations(range(len(input_list)), 2):
            for index_i in range(input_list[i]):
                for index_j in range(input_list[j]):
                    var_i = Var(set=i, index=index_i)
                    var_j = Var(set=j, index=index_j)
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
        1) 1) x_i + x_j - y_ij <= 1
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
            var_columns = (self.base_vars[Var(set=level, index=i)] for i in range(size))
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
        1) x_i + x_j - y_ij <= 1
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

