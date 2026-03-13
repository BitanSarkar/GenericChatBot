from pathlib import Path
import json
import pypdf
import pathspec

ROOT = Path(__file__).parent.parent
REFS_DIR   = ROOT / "refs"
OUTPUT_DIR = ROOT / "output"

TEXT_EXTENSIONS = {".txt", ".py", ".java", ".ts", ".js", ".html", ".css", ".xml", ".json", ".md", ".yaml", ".yml"}
PDF_EXTENSIONS  = {".pdf"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf_file(path: Path) -> tuple[str, str | None]:
    """
    Returns (text, skip_reason).
    skip_reason is None on success, or a string explaining why it failed.
    """
    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as e:
        return "", f"could not open PDF: {e}"

    if reader.is_encrypted:
        return "", "password protected"

    pages = [page.extract_text() for page in reader.pages if page.extract_text()]

    if not pages:
        return "", f"scanned PDF (no text layer) — {len(reader.pages)} pages, all images"

    return "\n".join(pages), None


def load_file(path: Path) -> tuple[dict | None, str | None]:
    """
    Returns (document, skip_reason).
    skip_reason is None on success, or a string explaining why it was skipped.
    """
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return None, f"unsupported extension '{ext}'"

    if ext in TEXT_EXTENSIONS:
        text       = load_text_file(path)
        skip_reason = None
    else:
        text, skip_reason = load_pdf_file(path)

    if skip_reason:
        return None, skip_reason

    if not text or not text.strip():
        return None, "empty file"

    return {"source": str(path.relative_to(ROOT)), "extension": ext, "text": text.strip()}, None


def load_refignore(base: Path) -> pathspec.PathSpec:
    refignore_path = base / ".refignore"
    if not refignore_path.exists():
        return pathspec.PathSpec.from_lines("gitwildmatch", [])
    patterns = refignore_path.read_text().splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def load_all() -> list[dict]:
    if not REFS_DIR.exists():
        raise FileNotFoundError(f"'{REFS_DIR}' not found. Create it and add your documents.")

    ignore_spec = load_refignore(REFS_DIR)
    documents   = []

    for path in sorted(REFS_DIR.rglob("*")):
        if not path.is_file():
            continue

        if ignore_spec.match_file(str(path.relative_to(REFS_DIR))):
            print(f"  Ignored : {path.relative_to(ROOT)}  (matched .refignore)")
            continue

        doc, reason = load_file(path)
        if doc:
            documents.append(doc)
            print(f"  Loaded  : {doc['source']}  ({len(doc['text'])} chars)")
        else:
            print(f"  Skipped : {path.relative_to(ROOT)}  ({reason})")

    return documents


if __name__ == "__main__":
    output_path = OUTPUT_DIR / "documents.json"
    output_path.parent.mkdir(exist_ok=True)

    print("Loading documents from refs/...\n")
    docs = load_all()

    output_path.write_text(json.dumps(docs, indent=2))
    print(f"\nTotal documents loaded: {len(docs)}")
    print(f"Saved to: {output_path.relative_to(ROOT)}")
