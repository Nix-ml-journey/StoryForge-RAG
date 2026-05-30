import argparse
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

    root = Path(__file__).resolve().parents[1]

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None  # type: ignore[assignment]

    # Importable implementation lives in the package.
    import sys

    sys.path.insert(0, str(root / "src"))
    from storyforge.data.step1_prepare_and_enrich import (  # type: ignore[import-not-found]
        run_step1_prepare_and_enrich,
    )

    def _run() -> None:
        run_step1_prepare_and_enrich(
            root=root,
            limit=args.limit,
            overwrite_summary=args.overwrite_summary,
            overwrite_sections=args.overwrite_sections,
            dry_run=args.dry_run,
            enable_tqdm=True,
        )

    if tqdm is None:
        run_step1_prepare_and_enrich(
            root=root,
            limit=args.limit,
            overwrite_summary=args.overwrite_summary,
            overwrite_sections=args.overwrite_sections,
            dry_run=args.dry_run,
            enable_tqdm=False,
        )
        return

    with tqdm(total=2, desc="Step 1", unit="step") as pbar:
        _run()
        pbar.update(2)


if __name__ == "__main__":
    main()

