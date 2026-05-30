"""
Compatibility shim for older imports.

The canonical config implementation lives in `storyforge.config.config`.
Keeping this module under `src/` ensures it is importable when the test
environment only adds `src` to `sys.path`.
"""

from storyforge.config.config import *  # noqa: F403

