import os
import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from os.path import isfile
from signal import signal, SIGCHLD, SIGALRM, alarm

from dicewars.ai.xgalba03.recording_driver import RecordingDriver
from dicewars.ai.xgalba03.recording_server import ServerRecord
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
# parser.add_argument('--ai', help="Specify AI versions as a sequence of ints.", nargs='+')
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


def clear_previous_data_if_exists():
    if isfile(ServerRecord.GAMES_FILE):
        os.remove(ServerRecord.GAMES_FILE)
    if isfile(RecordingDriver.BOARD_FILE):
        os.remove(RecordingDriver.BOARD_FILE)


def save_iteration(iteration):
    file = open(iteration_file, "w")
    file.write(str(iteration))


def get_iteration() -> int:
    iteration = 0
    if isfile(iteration_file):
        file = open(iteration_file, "r")
        iteration = int(file.readline())

    return iteration


ai_list = [
    "dt.rand",
    "dt.sdc",
    "dt.ste",
    "dt.stei",
    "dt.wpm_c",
    "dt.wpm_d",
    "dt.wpm_s",
    "kb.xlogin42",
    "kb.sdc_post_at",
    "kb.sdc_post_dt",
    "kb.sdc_pre_at",
    "kb.stei_adt",
    "kb.stei_at",
    "kb.stei_dt",
    "kb.xlogin00"
]

training = "xgalba03"
iteration_file = "logs/iteration.txt"

if __name__ == "__main__":
    args = parser.parse_args()
    signal(SIGCHLD, signal_handler)

    clear_previous_data_if_exists()
    iteration = get_iteration()

    for i in range(args.nb_games):
        ai = [training]
        enemy_count = random.randint(1, 3)
        for _ in range(enemy_count):
            index = random.randint(0, len(ai_list) - 1)
            while ai_list[index] in ai:
                index = random.randint(0, len(ai_list) - 1)
            ai.append(ai_list[index])
        random.shuffle(ai)

        if args.report:
            print(20 * "=" + " game: {} iteration: {} ".format(i, iteration) + 20 * "=")
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time)
            sys.stdout.write('playing AIs: {}\n'.format(ai))
        try:
            board_seed = None if args.board is None else args.board + i
            board_definition = BoardDefinition(board_seed, args.ownership, args.strength)
            run_ai_only_game(
                args.port, args.address, procs, ai,
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
        iteration += 1
        save_iteration(iteration)
        print(40 * "=")
