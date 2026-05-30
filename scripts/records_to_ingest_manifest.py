def main() -> None:
    from storyforge.data.records_to_manifest import records_to_ingest_manifest  # type: ignore[import-not-found]

    res = records_to_ingest_manifest()
    print(f"Wrote {res.get('written', 0)} chunks to {res.get('out_path')} from records in {res.get('records_dir')}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "src")
    main()

