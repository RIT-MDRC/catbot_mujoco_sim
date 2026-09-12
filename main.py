"""Launch the Catbot Marimo notebook with file watching enabled.
Run `uv run noteboook` to launch the notebook
"""

from __future__ import annotations

import os


def main() -> None:
    """Replace this process with the watched Marimo editor."""
    os.execvp("uv", ["uv", "run", "marimo", "edit", "notebook/main.py", "--watch"])


if __name__ == "__main__":
    main()
