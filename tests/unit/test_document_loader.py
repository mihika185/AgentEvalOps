from pathlib import Path

import pytest
from backend.app.ingestion.document_loader import DocumentLoadError, load_document

def write_simple_pdf(path: Path, lines: list[str]) -> None:
    def escape_pdf_text(text: str) -> str:
        return (
            text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )

    content_lines = [
        "BT",
        "/F1 12 Tf",
        "50 760 Td",
        "16 TL",
    ]

    for line in lines:
        content_lines.append(f"({escape_pdf_text(line)}) Tj")
        content_lines.append("T*")

    content_lines.append("ET")

    content = "\n".join(content_lines).encode("utf-8")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(content)).encode("utf-8")
        + b" >>\nstream\n"
        + content
        + b"\nendstream",
    ]

    pdf = bytearray()
    pdf.extend(b"%PDF-1.4\n")

    offsets = []

    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("utf-8"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)

    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("utf-8"))
    pdf.extend(b"0000000000 65535 f \n")

    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("utf-8"))

    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n".encode("utf-8")
    )

    path.write_bytes(pdf)

def test_load_markdown_document(tmp_path):
    file_path = tmp_path / "policy.md"
    file_path.write_text(
        "# Refund Policy\n\nCustomers can request a refund within 7 days.",
        encoding="utf-8"
    )

    document = load_document(file_path)

    assert document.filename == "policy.md"
    assert document.file_type == "md"
    assert "refund within 7 days" in document.text
    assert document.num_pages is None
    assert document.metadata["loader"] == "plain_text_loader"

def test_load_empty_text_document_fails(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="File is empty"):
        load_document(file_path)

def test_load_unsupported_file_type_fails(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("name,value\nabc,123", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="Unsupported file type"):
        load_document(file_path)

def test_load_pdf_document(tmp_path):
    file_path = tmp_path / "policy.pdf"

    write_simple_pdf(
        file_path,
        [
            "Company Refund Policy",
            "Customers can request a refund within 7 days.",
            "Subscription cancellations take effect at the end of the billing cycle.",
        ]
    )

    document = load_document(file_path)

    assert document.filename == "policy.pdf"
    assert document.file_type == "pdf"
    assert document.num_pages == 1
    assert document.metadata["loader"] == "pypdf_loader"
    assert document.metadata["ocr_used"] is False
    assert "Subscription cancellations" in document.text