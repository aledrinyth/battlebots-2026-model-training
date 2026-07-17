"""Match-legal bot package.

Hard rule (enforced by import-linter, contract in ``.importlinter``): nothing
under ``bot/`` may import ``train`` or ``torch``. The bot must run inside the
grader sandbox whose only extra library beyond the kit's pydantic/dotmap is
numpy (networkx / pandas / wasmtime are whitelisted but deliberately unused).

Keep this module import-light: importing ``bot`` must succeed in the bot venv
(``requirements-bot.txt``), which does NOT contain torch.
"""

__all__: list[str] = []
