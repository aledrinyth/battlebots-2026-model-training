#!/usr/bin/env python3

from __future__ import annotations

import argparse
from importlib.resources import files
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from signal import SIGKILL

from lib.config.arena import NUM_PLAYERS


PIPE_PERMISSIONS = 0o660
DIRECTORY_PERMISSIONS = 0o775
START_DELAY_SECONDS = 3.0


def default_submission_path() -> Path:
    return Path(str(files("agario_visualiser.examples").joinpath("example.py"))).resolve()


def parse_args(mode: str) -> argparse.Namespace:
    default_workspace = Path(".agario") / mode
    expected = NUM_PLAYERS - 1 if mode == "interactive" else NUM_PLAYERS
    parser = argparse.ArgumentParser(
        description=(
            "Run a local Agar.io match in interactive mode."
            if mode == "interactive"
            else "Run a local Agar.io match in simulation mode."
        )
    )
    parser.add_argument(
        "submission",
        nargs="*",
        help=(
            "Submission specs in count:path form. "
            f"The counts must sum to {expected} for this {NUM_PLAYERS}-player mode."
        ),
    )
    parser.add_argument(
        "--visualiser-player",
        type=int,
        default=0,
        help="Submission slot to reserve for the visualiser.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the visualiser submission without opening a GUI.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=720,
        help="Visualiser window size in pixels.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=default_workspace,
        help="Directory to use for match pipes, logs, and engine outputs.",
    )
    return parser.parse_args()


def parse_submission_specs(specs: list[str], expected_total: int) -> list[Path]:
    default_bot = default_submission_path()
    if not specs:
        return [default_bot] * expected_total

    expanded: list[Path] = []
    for spec in specs:
        count_str, delim, path_str = spec.partition(":")
        if delim == "" or path_str == "":
            raise SystemExit(
                f"Invalid submission spec '{spec}'. Expected count:path, for example 2:bots/my_bot.py."
            )
        try:
            count = int(count_str)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid submission count in '{spec}'. Counts must be positive integers."
            ) from exc
        if count <= 0:
            raise SystemExit(
                f"Invalid submission count in '{spec}'. Counts must be positive integers."
            )
        expanded.extend([Path(path_str).resolve()] * count)

    if len(expanded) != expected_total:
        raise SystemExit(
            f"Submission counts must sum to {expected_total} for this {NUM_PLAYERS}-player mode."
        )
    return expanded


def build_submission_plan(
    args: argparse.Namespace,
    mode: str,
) -> tuple[list[Path | None], Path | None]:
    if args.visualiser_player < 0 or args.visualiser_player >= NUM_PLAYERS:
        raise SystemExit(
            f"--visualiser-player must be between 0 and {NUM_PLAYERS - 1}."
        )

    interactive = mode == "interactive"
    needed = NUM_PLAYERS - 1 if interactive else NUM_PLAYERS
    bot_paths = parse_submission_specs(args.submission, needed)

    resolved: list[Path | None] = []
    bot_iter = iter(bot_paths)
    delegated_script: Path | None = None
    for player in range(NUM_PLAYERS):
        if player == args.visualiser_player:
            resolved.append(None)
            if not interactive:
                delegated_script = next(bot_iter)
        else:
            resolved.append(next(bot_iter))
    return resolved, delegated_script


def clean_environment_for_player(workspace_root: Path, player: int) -> None:
    shutil.rmtree(workspace_root / f"submission{player}", ignore_errors=True)


def setup_environment_for_player(workspace_root: Path, player: int) -> None:
    submission_dir = workspace_root / f"submission{player}" / "io"
    submission_dir.mkdir(parents=True, mode=DIRECTORY_PERMISSIONS, exist_ok=True)
    os.mkfifo(submission_dir / "to_engine.pipe", mode=PIPE_PERMISSIONS)
    os.mkfifo(submission_dir / "from_engine.pipe", mode=PIPE_PERMISSIONS)


def setup_match_environment(workspace_root: Path) -> None:
    workspace_root = workspace_root.resolve()
    workspace_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(workspace_root, ignore_errors=True)
    workspace_root.mkdir(parents=True)
    (workspace_root / "output").mkdir()
    (workspace_root / "input").mkdir()

    for player in range(NUM_PLAYERS):
        clean_environment_for_player(workspace_root, player)
        setup_environment_for_player(workspace_root, player)

    catalog = [{"team_id": i} for i in range(NUM_PLAYERS)]
    with open(workspace_root / "input" / "catalog.json", "w") as file:
        file.write(json.dumps(catalog))


def runtime_env(workspace_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["GAME_ENGINE_CORE_DIRECTORY"] = str(workspace_root.resolve())
    return env


def start_submissions(
    workspace_root: Path,
    submission_paths: list[Path | None],
    args: argparse.Namespace,
    delegated_script: Path | None,
    interactive: bool,
) -> list[tuple[int, bool]]:
    env = runtime_env(workspace_root)
    processes: list[subprocess.Popen[str]] = []
    is_visualiser_process: list[bool] = []

    for player, script_path in enumerate(submission_paths):
        submission_root = workspace_root / f"submission{player}"
        stdout_path = submission_root / "io" / "submission.log"
        stderr_path = submission_root / "io" / "submission.err"
        with open(stdout_path, "w") as stdout_file, open(stderr_path, "w") as stderr_file:
            if script_path is None:
                command = [sys.executable, "-m", "agario_visualiser.visualiser_submission"]
                is_visualiser = True
                if interactive:
                    command.append("--masquerade")
                elif delegated_script is not None:
                    command.extend(["--delegate-script", str(delegated_script)])
                if args.headless:
                    command.append("--headless")
                command.extend(["--countdown-seconds", str(START_DELAY_SECONDS)])
                command.extend(["--window-size", str(args.window_size)])
            else:
                command = [sys.executable, str(script_path)]
                is_visualiser = False

            process = subprocess.Popen(
                command,
                cwd=submission_root,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )
        processes.append(process)
        is_visualiser_process.append(is_visualiser)
        print(f"[visualiser-launcher] started submission {player} (pid={process.pid}).")

    return [
        (process.pid, is_visualiser)
        for process, is_visualiser in zip(processes, is_visualiser_process, strict=True)
    ]


def start_engine(workspace_root: Path, realtime: bool) -> None:
    env = runtime_env(workspace_root)
    print("[visualiser-launcher] started engine.")
    with open(workspace_root / "output" / "engine.log", "w") as stdout_file, open(
        workspace_root / "output" / "engine.err", "w"
    ) as stderr_file:
        command = [sys.executable, "-m", "engine"]
        if realtime:
            command.append("--realtime")
        process = subprocess.Popen(
            command,
            cwd=workspace_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            universal_newlines=True,
            bufsize=1,
        )

        while True:
            if process.stdout is None:
                break
            data = process.stdout.read(1)
            if not data:
                break
            print(data, end="", flush=True)
            stdout_file.write(data)

    print("[visualiser-launcher] engine terminated.")


def run_mode(mode: str) -> None:
    args = parse_args(mode)
    workspace_root = args.workspace.resolve()
    interactive = mode == "interactive"
    submission_paths, delegated_script = build_submission_plan(args, mode)
    setup_match_environment(workspace_root)
    pids = start_submissions(
        workspace_root,
        submission_paths,
        args,
        delegated_script,
        interactive,
    )

    try:
        print(
            f"[visualiser-launcher] delaying engine start for {START_DELAY_SECONDS:.0f}s countdown."
        )
        time.sleep(START_DELAY_SECONDS)
        start_engine(workspace_root, realtime=not args.headless)
    finally:
        for pid, is_visualiser in pids:
            if is_visualiser:
                print(
                    f"[visualiser-launcher] leaving visualiser pid {pid} running for the end screen."
                )
                continue
            try:
                os.kill(pid, SIGKILL)
                print(f"[visualiser-launcher] terminated submission pid {pid}.")
            except ProcessLookupError:
                pass


def interactive_main() -> None:
    run_mode("interactive")


def simulation_main() -> None:
    run_mode("simulation")


def main() -> None:
    simulation_main()


if __name__ == "__main__":
    main()
