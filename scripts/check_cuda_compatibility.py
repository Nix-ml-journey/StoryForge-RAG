"""
CLI wrapper for `storyforge.scripts.check_cuda_compatibility`.

Run:
  py scripts/check_cuda_compatibility.py
"""

from storyforge.scripts.check_cuda_compatibility import run_cuda_compatibility_check


if __name__ == "__main__":
    run_cuda_compatibility_check()
