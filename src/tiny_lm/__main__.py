"""Support ``python -m tiny_lm`` without running the CLI on import."""

from tiny_lm.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
