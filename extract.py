from __future__ import annotations

import time
import uuid
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
    granite_picture_description,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ContentLayer, ImageRefMode
from docling_core.types.doc.document import DoclingDocument

import config

LAYERS = {ContentLayer.BODY, ContentLayer.FURNITURE}


def stable_doc_id(doc: DoclingDocument) -> str:
    if doc.origin is None:
        raise ValueError("document has no origin; cannot derive a doc_id")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(doc.origin.binary_hash)))


def build_converter(describe_pictures: bool, ocr_lang: str | None = None) -> DocumentConverter:
    options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        ocr_options=RapidOcrOptions(backend="onnxruntime", lang=[ocr_lang or config.OCR_LANG]),
        do_table_structure=True,
        images_scale=1,
        generate_picture_images=describe_pictures,
        do_picture_description=describe_pictures,
        do_picture_classification=describe_pictures,
        generate_parsed_pages=True,
    )
    options.table_structure_options.mode = TableFormerMode.FAST
    options.table_structure_options.do_cell_matching = True
    options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)

    if describe_pictures:
        options.picture_description_options = granite_picture_description

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def parse_pages(spec: str) -> tuple[int, int]:
    start, _, end = spec.partition("-")
    try:
        first, last = int(start), int(end) if end else int(start)
    except ValueError:
        raise SystemExit(f"error: bad --pages {spec!r} (expected N or N-M)")
    if first < 1 or last < first:
        raise SystemExit(f"error: bad --pages {spec!r} (expected N or N-M)")
    return first, last


def extract(
    source: str,
    describe_pictures: bool = False,
    pages: tuple[int, int] | None = None,
    ocr_lang: str | None = None,
) -> DoclingDocument:
    if not source.startswith(("http://", "https://")):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        source = str(path.resolve())

    print(f"Converting {source} ...")
    started = time.perf_counter()
    converter = build_converter(describe_pictures, ocr_lang)
    result = converter.convert(source, **({"page_range": pages} if pages else {}))
    doc = result.document
    print(f"Converted in {time.perf_counter() - started:.1f}s")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.OUTPUT_DIR / f"{stable_doc_id(doc)}.json"
    doc.save_as_json(json_path, image_mode=ImageRefMode.PLACEHOLDER, ensure_ascii=False)
    doc.save_as_markdown(
        json_path.with_suffix(".md"),
        image_mode=ImageRefMode.PLACEHOLDER,
        included_content_layers=LAYERS,
    )
    print(f"Saved {json_path} and {json_path.with_suffix('.md')}")
    return doc