import argparse
import sys

from engine.game_engine import GameEngine


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Agar.io engine.")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Throttle turns to TURN_DURATION_SECONDS in realtime modes.",
    )
    parser.add_argument(
        "--print-recording-interactive",
        action="store_true",
        help="Legacy alias for --realtime.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    game = GameEngine(realtime=args.realtime or args.print_recording_interactive)
    game.start()


if __name__ == "__main__":
    main()
