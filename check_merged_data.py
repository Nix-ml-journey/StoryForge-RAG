"""
Check that Stories (.txt), Metadata (.json), and Data_Merged (.json) are in sync:
- Same set of names (stems) across all three folders
- Merged JSON "documents" matches the corresponding .txt content
- Merged JSON metadata matches the corresponding Metadata JSON (excluding documents)
"""

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    if yaml is None:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_stems(folder: Path, ext: str) -> set[str]:
    """Return set of file stems (name without extension) in folder."""
    if not folder.exists():
        return set()
    return {f.stem for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ext}


def read_txt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metadata_without_documents(data: dict) -> dict:
    """Return a copy of the dict without 'documents' key (for comparison)."""
    return {k: v for k, v in data.items() if k != "documents"}


def run_checks(
    base_path: Path,
    story_folder_name: str,
    metadata_folder_name: str,
    merged_folder_name: str,
    verbose: bool = False,
) -> bool:
    story_dir = base_path / story_folder_name
    meta_dir = base_path / metadata_folder_name
    merged_dir = base_path / merged_folder_name

    stories = get_stems(story_dir, ".txt")
    metadata = get_stems(meta_dir, ".json")
    merged = get_stems(merged_dir, ".json")

    all_ok = True

    # ---- 1) Same set of stems ----
    in_all_three = stories & metadata & merged
    only_stories = stories - metadata - merged
    only_metadata = metadata - stories - merged
    only_merged = merged - stories - metadata
    missing_merged = (stories & metadata) - merged
    missing_story = (metadata & merged) - stories
    missing_metadata = (stories & merged) - metadata

    def report(msg: str, items: set, is_error: bool = True):
        nonlocal all_ok
        if not items:
            return
        if is_error:
            all_ok = False
        print(msg)
        for s in sorted(items):
            print(f"  - {s}")

    print("=" * 60)
    print("CHECK: Same stems across Stories, Metadata, Data_Merged")
    print("=" * 60)

    report("Only in Stories (no matching Metadata or Merged):", only_stories)
    report("Only in Metadata (no matching Story or Merged):", only_metadata)
    report("Only in Data_Merged (no matching Story or Metadata):", only_merged)
    report("In Stories + Metadata but missing in Data_Merged:", missing_merged)
    report("In Metadata + Data_Merged but missing Story .txt:", missing_story)
    report("In Stories + Data_Merged but missing Metadata .json:", missing_metadata)

    if in_all_three:
        print(f"\nIn sync (in all three): {len(in_all_three)} items")
        if verbose:
            for s in sorted(in_all_three):
                print(f"  - {s}")

    # ---- 2) Content checks for stems in all three ----
    if not in_all_three:
        print("\nSkipping content checks (no stem in all three folders).")
        return all_ok

    print("\n" + "=" * 60)
    print("CHECK: Merged 'documents' matches Story .txt content")
    print("=" * 60)

    docs_mismatch = []
    for stem in sorted(in_all_three):
        txt_path = story_dir / f"{stem}.txt"
        merged_path = merged_dir / f"{stem}.json"
        try:
            story_content = read_txt(txt_path)
            merged_data = read_json(merged_path)
            merged_docs = merged_data.get("documents", "")
            if story_content != merged_docs:
                docs_mismatch.append(stem)
                all_ok = False
        except Exception as e:
            print(f"  Error reading {stem}: {e}")
            all_ok = False

    if docs_mismatch:
        print("Merged 'documents' does NOT match .txt content:")
        for s in docs_mismatch:
            print(f"  - {s}")
    else:
        print("OK: All merged 'documents' match the corresponding .txt files.")

    print("\n" + "=" * 60)
    print("CHECK: Merged metadata matches Metadata JSON (ids, metadatas, chapters, series)")
    print("=" * 60)

    meta_mismatch = []
    for stem in sorted(in_all_three):
        meta_path = meta_dir / f"{stem}.json"
        merged_path = merged_dir / f"{stem}.json"
        try:
            meta_data = read_json(meta_path)
            merged_data = read_json(merged_path)
            # Metadata JSON may have "documents":""; merged has full text. Compare rest.
            meta_core = metadata_without_documents(meta_data)
            merged_core = metadata_without_documents(merged_data)
            if meta_core != merged_core:
                meta_mismatch.append(stem)
                all_ok = False
        except Exception as e:
            print(f"  Error comparing {stem}: {e}")
            all_ok = False

    if meta_mismatch:
        print("Merged metadata does NOT match Metadata JSON:")
        for s in meta_mismatch:
            print(f"  - {s}")
    else:
        print("OK: All merged metadata matches the corresponding Metadata JSON.")

    return all_ok


def main():
    root = Path(__file__).resolve().parent
    config_path = root / "setup.yaml"
    config = load_config(config_path)

    base_path = Path(config.get("BASE_PATH", root))
    story_input = config.get("Story_input", "Stories")
    metadata_input = config.get("Metadata_input", "Metadata")
    merged_output = config.get("Merged_data_output", "Data_Merged")

    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print(f"Base path: {base_path}")
    print(f"Stories:   {base_path / story_input}")
    print(f"Metadata:  {base_path / metadata_input}")
    print(f"Merged:   {base_path / merged_output}\n")

    ok = run_checks(base_path, story_input, metadata_input, merged_output, verbose=verbose)

    print("\n" + "=" * 60)
    if ok:
        print("Result: All checks passed.")
    else:
        print("Result: Some checks failed.")
    print("=" * 60)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
