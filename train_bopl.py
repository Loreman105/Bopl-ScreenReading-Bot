"""Backwards-compatible entrypoint.

This file stays runnable (`python train_bopl.py`) but the implementation
now lives in the `bopl_bot` package.
"""

from bopl_bot.train import train


def main() -> None:
    train()

if __name__ == "__main__":
    main()
 