def main() -> None:
    import argparse
    from storyforge.data.story_records import create_story_records  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data/story_json/*.json")
    parser.add_argument("--only", type=str, default="", help="Only generate one story record by filename stem")
    args = parser.parse_args()

    res = create_story_records(overwrite=args.overwrite, only=args.only)
    print(
        f"Story records: created={res.get('created', 0)}, "
        f"skipped_existing={res.get('skipped_existing', 0)}, folder={res.get('folder')}"
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "src")
    main()

