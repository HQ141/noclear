import csv
import os
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

from paddleocr import PaddleOCRVL

PDF_PATH = Path('/test.pdf')
OUTPUT_DIR = Path(os.environ.get('NOCLEAR_OUTPUT_DIR', 'output'))
# 'gpu': VL model runs in the vLLM server at VL_SERVER_URL. 'cpu': everything runs on the CPU.
DEVICE = os.environ.get('NOCLEAR_DEVICE', 'gpu').lower()
VL_SERVER_URL = os.environ.get('NOCLEAR_VL_SERVER_URL', 'http://127.0.0.1:8118/v1')

TABLE_HEADER = [
    'S#',
    'Standard Weight (Kg)',
    'Loading value (Kg)',
    'Unloading Value (Kg)',
    'Average Error (%)',
]

CALIBRATION_RE = re.compile(r'CALIBRATION\s*DATA\s*OF\s*[:\-]?\s*(.*)', re.IGNORECASE | re.DOTALL)
PAGE_SUFFIX_RE = re.compile(r'\bPage\s*#?.*$', re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r'\d+(?:\.\d+)?')


class _TableParser(HTMLParser):
    """Collects the text of each <td>/<th> cell, row by row."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self._row = []
        elif tag in ('td', 'th') and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self._row is not None and self._cell is not None:
            self._row.append(' '.join(''.join(self._cell).split()))
            self._cell = None
        elif tag == 'tr' and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_table(content: str) -> list[list[str]]:
    """Turn a table block (HTML, or a markdown pipe table as fallback) into rows of cells."""
    if '<table' in content.lower():
        parser = _TableParser()
        parser.feed(content)
        return parser.rows
    rows = []
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith('|') or set(line) <= set('|-: '):
            continue
        rows.append([cell.strip() for cell in line.strip('|').split('|')])
    return rows


def clean_value(text: str) -> str:
    """Keep the number from a handwritten cell, dropping tick marks and units ("50kg" -> "50")."""
    match = NUMBER_RE.search(text)
    return match.group() if match else text.strip()


def find_calibration_of(blocks) -> str:
    """Return the handwritten value after "CALIBRATION DATA OF" (e.g. "Aggregate")."""
    texts = [b.content for b in blocks if b.label != 'table' and b.content]
    for i, text in enumerate(texts):
        match = CALIBRATION_RE.search(text)
        if not match:
            continue
        value = match.group(1)
        # The handwriting is sometimes detected as its own block right after the label.
        if not PAGE_SUFFIX_RE.sub('', value).strip(' _:-\n') and i + 1 < len(texts):
            value = texts[i + 1]
        return PAGE_SUFFIX_RE.sub('', value).strip(' _:-\n')
    return ''


def average_error(standard: str, loading: str, unloading: str) -> str:
    """Mean of the loading and unloading errors relative to the standard weight, e.g. "0.21%"."""
    try:
        std, load, unload = float(standard), float(loading), float(unloading)
    except ValueError:
        return ''
    if std == 0:
        return ''
    error = (abs(load - std) + abs(unload - std)) / 2 / std * 100
    return f'{error:.2f}%'


def page_rows(blocks) -> list[list[str]]:
    rows = []
    for block in blocks:
        if block.label != 'table':
            continue
        for cells in parse_table(block.content):
            if not cells or not cells[0].strip().isdigit():
                continue  # header row, or a row without an S# number
            cells = (cells + [''] * 4)[:4]
            standard, loading, unloading = (clean_value(c) for c in cells[1:4])
            if not (standard or loading or unloading):
                continue  # empty/crossed-out rows at the bottom of the form
            rows.append(
                [
                    cells[0].strip(),
                    standard,
                    loading,
                    unloading,
                    average_error(standard, loading, unloading),
                ]
            )
    return rows


CSV_HEADER = ['Page', 'Calibration Data of', *TABLE_HEADER]


def extract_rows(pdf_path: Path, pipeline: PaddleOCRVL, on_page=None) -> list[list]:
    """OCR every page of the PDF and return the CSV rows (without the header).

    on_page(page_number, page_count), if given, is called after each page finishes.
    """
    rows = []
    # predict_iter yields each page as it finishes (predict() waits for the whole PDF).
    for res in pipeline.predict_iter(str(pdf_path)):
        page_number = (res['page_index'] or 0) + 1
        blocks = res['parsing_res_list']
        calibration_of = find_calibration_of(blocks)
        page = page_rows(blocks)
        print(f'{pdf_path.name} page {page_number}: {calibration_of!r}, {len(page)} rows')
        rows.extend([page_number, calibration_of, *row] for row in page)
        if on_page:
            on_page(page_number, res['page_count'])
    return rows


def write_csv(rows: list[list], csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    return csv_path


def extract(pdf_path: Path, pipeline: PaddleOCRVL) -> Path:
    rows = extract_rows(pdf_path, pipeline)
    return write_csv(rows, OUTPUT_DIR / f'{pdf_path.stem}.csv')


def create_pipeline() -> PaddleOCRVL:
    if DEVICE == 'cpu':
        # CPU only: both the layout and VL models run in PaddlePaddle on the CPU
        print('VL model on: CPU')
        return PaddleOCRVL(pipeline_version='v1', device='cpu')
    # GPU: VL model runs in vLLM on the GPU (needs the vLLM server running)
    print(f'VL model on: GPU via {VL_SERVER_URL}')
    return PaddleOCRVL(
        pipeline_version='v1',  # matches PaddlePaddle/PaddleOCR-VL served as PaddleOCR-VL-0.9B
        device='cpu',  # layout model only; it's small and PaddlePaddle can't use AMD GPUs
        vl_rec_backend='vllm-server',
        vl_rec_server_url=VL_SERVER_URL,
    )


def main() -> None:
    pdf_paths = [Path(p) for p in sys.argv[1:]] or [PDF_PATH]
    pipeline = create_pipeline()
    for pdf_path in pdf_paths:
        start = time.perf_counter()
        csv_path = extract(pdf_path, pipeline)
        print(f'Wrote {csv_path} in {time.perf_counter() - start:.1f}s')
