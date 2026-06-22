from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.config import settings
from backend.app.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}

class DocumentLoadError(Exception):
    pass

@dataclass(frozen=True)
class LoadedDocument:
    filename: str
    file_type: str
    text: str
    metadata: dict[str, Any]

    @property
    def character_count(self) -> int:
        return len(self.text)


def get_file_type(file_path: Path) -> str:
    extension = file_path.suffix.lower()

    if extension == ".txt":
        return "txt"

    if extension in {".md", ".markdown"}:
        return "md"

    raise DocumentLoadError(f"Unsupported file type: {extension}")

def validate_file_path(file_path: Path) -> None:
    if not file_path.exists():
        raise DocumentLoadError(f"File does not exist: {file_path}")

    if not file_path.is_file():
        raise DocumentLoadError(f"Path is not a file: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported file type '{file_path.suffix}'. "
            f"Supported types are: {', '.join(sorted(SUPPORTED_TEXT_EXTENSIONS))}"
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
        metadata=metadata
    )

    logger.info(
        "Loaded document: %s (%s characters)",
        loaded_document.filename,
        loaded_document.character_count
    )

    return loaded_document