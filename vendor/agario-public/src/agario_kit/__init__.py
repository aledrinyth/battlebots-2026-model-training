"""Convenience meta-package for the public Agar.io stack."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib


def _read_version_from_pyproject() -> str:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as file:
        pyproject = tomllib.load(file)
    return str(pyproject["project"]["version"])


def get_engine_version() -> str:
    try:
        return version("agario-kit")
    except PackageNotFoundError:
        return _read_version_from_pyproject()


__version__ = get_engine_version()
