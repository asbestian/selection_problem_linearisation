import logging

from argparse import ArgumentParser
from representation.hrep import Hrep
from enum import Enum

module_logger = logging.getLogger('main')

class Linearisation(Enum):
    COMPACT = 1
    EXTENDED = 2
    GLOVER_WOOLSEY = 3

cmd_parser = ArgumentParser(description='Export linear representation of the selection problem.')
cmd_parser.add_argument('--lin', type=int, choices=range(1, 4), required=True,
                        help='Specifies which linearisation should be used: 1-compact, 2-extended, 3-glover_woolsey')
cmd_parser.add_argument('--input', metavar='size', action='store', type=int, nargs='+', required=True,
                        help='Specifies the respective sizes of the input sets as a sequence of numbers.')


if __name__ == '__main__':
    cmd_args = cmd_parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Given input: {cmd_args.input}')
    if (length := len(cmd_args.input)) < 2:
        logging.error(f'Expected len(input) >= 2. Found len(input) = {length}.')
    hrep = Hrep(cmd_args.input)
    lin = Linearisation(cmd_args.lin)
    if lin == Linearisation.GLOVER_WOOLSEY:
        print(hrep.get_glover_woolsey_rep())
    elif lin == Linearisation.COMPACT:
        print(hrep.get_compact_rep())
    elif lin == Linearisation.EXTENDED:
        logging.info('Not implemented yet.')
    else:
        logging.error('Unexpected linearisation.')