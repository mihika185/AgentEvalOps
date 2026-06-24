from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from backend.app.config import settings
from backend.app.logging_config import get_logger


logger = get_logger(__name__)

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS.union(SUPPORTED_PDF_EXTENSIONS)


class DocumentLoadError(Exception):
    pass


@dataclass(frozen=True)
class LoadedDocument:
    filename: str
    file_type: str
    text: str
    metadata: dict[str, Any]
    num_pages: Optional[int] = None

    @property
    def character_count(self) -> int:
        return len(self.text)


def get_file_type(file_path: Path) -> str:
    extension = file_path.suffix.lower()

    if extension == ".txt":
        return "txt"

    if extension in {".md", ".markdown"}:
        return "md"

    if extension == ".pdf":
        return "pdf"

    raise DocumentLoadError(f"Unsupported file type: {extension}")


def validate_file_path(file_path: Path) -> None:
    if not file_path.exists():
        raise DocumentLoadError(f"File does not exist: {file_path}")

    if not file_path.is_file():
        raise DocumentLoadError(f"Path is not a file: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported file type '{file_path.suffix}'. "
            f"Supported types are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    max_size_bytes = settings.max_upload_mb * 1024 * 1024
    file_size_bytes = file_path.stat().st_size

    if file_size_bytes > max_size_bytes:
        raise DocumentLoadError(
            f"File is too large: {file_size_bytes} bytes. "
            f"Maximum allowed size is {max_size_bytes} bytes."
        )


def load_document(file_path: str | Path) -> LoadedDocument:
    path = Path(file_path).expanduser().resolve()

    validate_file_path(path)

    if path.suffix.lower() == ".pdf":
        return load_pdf_document(path)

    return load_text_document(path)


def load_text_document(path: Path) -> LoadedDocument:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(
            f"Could not read file as UTF-8 text: {path}"
        ) from exc

    if not text.strip():
        raise DocumentLoadError(f"File is empty: {path}")

    file_size_bytes = path.stat().st_size
    line_count = text.count("\n") + 1

    metadata = {
        "source_path": str(path),
        "extension": path.suffix.lower(),
        "file_size_bytes": file_size_bytes,
        "line_count": line_count,
        "character_count": len(text),
        "loader": "plain_text_loader"
    }

    loaded_document = LoadedDocument(
        filename=path.name,
        file_type=get_file_type(path),
        text=text,
        metadata=metadata,
        num_pages=None
    )

    logger.info(
        "Loaded document: %s (%s characters)",
        loaded_document.filename,
        loaded_document.character_count
    )

    return loaded_document


def load_pdf_document(path: Path) -> LoadedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentLoadError(
            "pypdf is not installed. Install it with: "
            "python3 -m pip install \"pypdf>=4.0.0,<6.0.0\""
        ) from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise DocumentLoadError(f"Could not read PDF file: {path}") from exc

    if reader.is_encrypted:
        try:
            decrypt_result = reader.decrypt("")
        except Exception as exc:
            raise DocumentLoadError(
                "PDF is encrypted and could not be decrypted"
            ) from exc

        if decrypt_result == 0:
            raise DocumentLoadError(
                "PDF is encrypted and requires a password"
            )

    page_texts: list[str] = []

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            logger.warning(
                "Could not extract text from PDF page %s in %s",
                page_index,
                path
            )
            page_text = ""

        cleaned_page_text = page_text.strip()

        if not cleaned_page_text:
            continue

        page_texts.append(
            f"[Page {page_index}]\n{cleaned_page_text}"
        )

    if not page_texts:
        raise DocumentLoadError(
            "PDF does not contain extractable text. "
            "Scanned/image-only PDFs are not supported yet."
        )

    text = "\n\n".join(page_texts)

    file_size_bytes = path.stat().st_size
    num_pages = len(reader.pages)

    metadata = {
        "source_path": str(path),
        "extension": path.suffix.lower(),
        "file_size_bytes": file_size_bytes,
        "num_pages": num_pages,
        "pages_with_text": len(page_texts),
        "line_count": text.count("\n") + 1,
        "character_count": len(text),
        "loader": "pypdf_loader",
        "ocr_used": False
    }

    loaded_document = LoadedDocument(
        filename=path.name,
        file_type="pdf",
        text=text,
        metadata=metadata,
        num_pages=num_pages
    )

    logger.info(
        "Loaded PDF document: %s (%s pages, %s characters)",
        loaded_document.filename,
        num_pages,
        loaded_document.character_count
    )

    return loaded_document