def main() -> None:
    import argparse

    from storyforge.data.enrich_records import enrich_story_records  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Only process N records (0 = all)")
    parser.add_argument("--overwrite-summary", action="store_true", help="Overwrite existing non-empty summary")
    parser.add_argument("--overwrite-sections", action="store_true", help="Overwrite existing non-empty section labels")
    parser.add_argument("--skip-summary", action="store_true", help="Do not generate summary")
    parser.add_argument("--skip-sections", action="store_true", help="Do not label chunk sections")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes")
    args = parser.parse_args()

    res = enrich_story_records(
        limit=args.limit,
        overwrite_summary=args.overwrite_summary,
        overwrite_sections=args.overwrite_sections,
        skip_summary=args.skip_summary,
        skip_sections=args.skip_sections,
        dry_run=args.dry_run,
    )
    print(f"Enriched records: updated={res.get('updated', 0)}, total={res.get('total', 0)}, dry_run={res.get('dry_run')}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    sys.path.insert(0, "src")
    main()

