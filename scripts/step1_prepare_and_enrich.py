import argparse
import sys
from pathlib import Path


def main() -> None:
    """
    One-command Step 1:
    - Create missing `data/story_json/*.json` from `data/stories/*.txt`
    - Enrich records with `summary` + per-chunk `section`
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Only process N records for enrichment (0 = all)")
    parser.add_argument("--overwrite-summary", action="store_true")
    parser.add_argument("--overwrite-sections", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from storyforge.data.step1_prepare_and_enrich import run_step1_prepare_and_enrich

    run_step1_prepare_and_enrich(
        limit=args.limit,
        overwrite_summary=args.overwrite_summary,
        overwrite_sections=args.overwrite_sections,
        dry_run=args.dry_run,
        enable_tqdm=True,
    )


if __name__ == "__main__":
    main()
