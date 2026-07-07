#!/usr/bin/env python

import json
import shutil
from signal import SIGKILL
import subprocess
import sys
import os
from typing import Tuple

from lib.config.arena import NUM_PLAYERS

PIPE_PERMISSIONS = 0o660
FILE_PERMISSIOSN = 0o664
DIRECTORY_PERMISSIONS = 0o775


def main():
    commands = parse_cmd_args(sys.argv[1:])

    try:
        sources = [
            (int(x.split(":")[0]), x.split(":")[1]) for x in commands["--submissions"]
        ]
    except (ValueError, KeyError):
        print_usage()

    if sum([x[0] for x in sources]) != NUM_PLAYERS:
        print(f"Total players in the match must be {NUM_PLAYERS}.")
        print_usage()

    setup_environments(sources)
    submission_pids = start_submissions()

    if "--engine" in commands:
        if len(commands["--engine"]) != 0:
            print_usage()
        start_engine()

    else:
        print(
            "Once you have finished running the engine, press [Enter] to terminate any still-running submission processes."
        )
        input()

    for pid in submission_pids:
        print(f"[simulator]: terminating submission pid {pid}.")
        os.kill(pid, SIGKILL)

    print("[simulator] simulation complete.")


def parse_cmd_args(args: list[str]):
    commands = {}

    current_command = None
    for arg in args:
        if arg[:2] == "--":
            current_command = arg
            commands[current_command] = []
            continue

        if current_command is None:
            print_usage()

        commands[current_command].append(arg)

    for command in commands.keys():
        if command not in ["--submissions", "--engine"]:
            print_usage()

    return commands


def print_usage():
    print(
        "Usage: python3 scripts/match_simulator.py --submissions <count>:<path> [<count>:<path> ...] [--engine]\n"
        "   options:\n"
        f"       --submissions <count>:<path> ...           Run <count> copies of the submission at <path>. The counts across all entries must add up to {NUM_PLAYERS}.\n"
        "       --engine                                    If present, the simulator will start the engine. Without this flag you must start the engine manually.\n"
        "\n"
        "   examples:\n"
        "       uv run python scripts/match_simulator.py --submissions 4:examples/submissions/cautious.py --engine\n"
        "       uv run python scripts/match_simulator.py --submissions 1:examples/submissions/cautious.py 3:examples/submissions/dont_move.py --engine\n"
        "       uv run python scripts/match_simulator.py --submissions 2:examples/submissions/cautious.py 2:examples/submissions/dont_move.py\n"
    )
    sys.exit(0)


def setup_environments(sources: list[Tuple[int, str]]):
    shutil.rmtree("output", ignore_errors=True)
    os.mkdir("output")
    shutil.rmtree("input", ignore_errors=True)
    os.mkdir("input")

    count = 0
    source = sources.pop(0)
    for player in range(NUM_PLAYERS):
        if count >= source[0]:
            count = 0
            source = sources.pop(0)

        clean_environment_for_player(player)
        setup_environment_for_player(player, source[1])

        count += 1

    catalog = [{"team_id": i} for i in range(NUM_PLAYERS)]
    with open("input/catalog.json", "w") as f:
        f.write(json.dumps(catalog))


def start_submissions() -> list[int]:
    player_pids = []
    for player in range(NUM_PLAYERS):
        os.chdir(f"submission{player}")

        with (
            open("io/submission.log", "w") as f_log,
            open("io/submission.err", "w") as f_err,
        ):
            process = subprocess.Popen(
                ["python3", "submission.py"], stdout=f_log, stderr=f_err
            )

        player_pids.append(process.pid)
        print(f"[simulator]: started submission {player} (pid={process.pid}).")
        os.chdir("..")

    return player_pids


def start_engine():
    print("[simulator] started engine.")
    with (
        open("output/engine.log", "w") as f_log,
        open("output/engine.err", "w") as f_err,
    ):
        process = subprocess.Popen(
            ["python3", "-m", "engine", "--realtime"],
            stdout=subprocess.PIPE,
            stderr=f_err,
            text=True,
            universal_newlines=True,
            bufsize=1,
        )

        while True:
            if process.stdout is not None:
                data = process.stdout.read(1)
                if not data:
                    break
                print(data, end="", flush=True)
                f_log.write(data)

    print("[simulator] engine terminated.")


def setup_environment_for_player(player: int, source: str):
    os.makedirs(f"submission{player}/io", mode=DIRECTORY_PERMISSIONS)
    os.mkfifo(f"submission{player}/io/to_engine.pipe", mode=PIPE_PERMISSIONS)
    os.mkfifo(f"submission{player}/io/from_engine.pipe", mode=PIPE_PERMISSIONS)
    shutil.copy(source, f"submission{player}/submission.py")


def clean_environment_for_player(player: int):
    shutil.rmtree(f"submission{player}", ignore_errors=True)


if __name__ == "__main__":
    main()
