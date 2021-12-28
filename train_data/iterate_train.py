import os
import sys
from argparse import ArgumentParser
from os.path import isfile
from signal import signal, SIGCHLD

from dicewars.ai.test.recording_server import ServerRecord
from scripts.utils import BoardDefinition, run_ai_only_game
from train_data.model_training import train

parser = ArgumentParser(prog='Dice_Wars')
parser.add_argument('-n', '--nb-games', help="Number of games.", type=int, default=1)
parser.add_argument('-p', '--port', help="Server port", type=int, default=5005)
parser.add_argument('-a', '--address', help="Server address", default='127.0.0.1')
parser.add_argument('-b', '--board', help="Seed for generating board", type=int)
parser.add_argument('-s', '--strength', help="Seed for dice assignment", type=int)
parser.add_argument('-o', '--ownership', help="Seed for province assignment", type=int)
parser.add_argument('-f', '--fixed', help="Random seed to be used for player order and dice rolls", type=int)
parser.add_argument('-c', '--client-seed', help="Seed for clients", type=int)
parser.add_argument('-l', '--logdir', help="Folder to store last running logs in.")
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('--ai', help="Specify AI versions as a sequence of ints.", nargs='+')
parser.add_argument('-r', '--report', help="State the game number on the stdout", action='store_true')

procs = []


def signal_handler(signum, frame):
    """Handler for SIGCHLD signal that terminates server and clients
    """
    for p in procs:
        try:
            p.kill()
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    args = parser.parse_args()
    signal(SIGCHLD, signal_handler)

    for i in range(args.nb_games):
        if args.report:
            sys.stdout.write('{}\n'.format(i))
        try:
            board_seed = None if args.board is None else args.board + i
            board_definition = BoardDefinition(board_seed, args.ownership, args.strength)
            run_ai_only_game(
                args.port, args.address, procs, args.ai,
                board_definition,
                fixed=args.fixed,
                client_seed=args.client_seed,
                logdir=args.logdir,
                debug=args.debug,
            )
        except KeyboardInterrupt:
            for p in procs:
                p.kill()
            break
        except AttributeError:
            for p in procs:
                p.kill()

        print("Training model")
        train()
