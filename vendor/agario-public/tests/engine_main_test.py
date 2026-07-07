from engine.__main__ import parse_args


def test_parse_args_defaults_to_non_realtime() -> None:
    args = parse_args([])

    assert not args.realtime
    assert not args.print_recording_interactive


def test_parse_args_accepts_realtime_flag() -> None:
    args = parse_args(["--realtime"])

    assert args.realtime


def test_parse_args_accepts_legacy_realtime_alias() -> None:
    args = parse_args(["--print-recording-interactive"])

    assert args.print_recording_interactive
