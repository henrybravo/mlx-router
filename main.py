#!/usr/bin/env python3
"""
Development entry point — delegates to mlx_router.cli.main().

Use `python main.py --config config.json` for local development.
When installed from PyPI, use the `mlx-router` command instead.
"""

from mlx_router.cli import main

if __name__ == "__main__":
    main()
