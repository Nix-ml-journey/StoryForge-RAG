"""StoryForge package."""

import logging as _logging

# Configured once here (imported by every entry point: main.py, scripts/, tests)
# instead of being re-declared with logging.basicConfig(...) in individual modules,
# where only the first call actually takes effect and the rest are dead code.
_logging.basicConfig(level=_logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
