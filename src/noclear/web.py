"""Web interface: upload a calibration PDF, watch progress, preview the table, download the CSV."""

import csv
import io
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from . import CSV_HEADER, OUTPUT_DIR, create_pipeline, extract_rows, write_csv

INDEX_HTML = Path(__file__).with_name('index.html')
UPLOAD_DIR = Path(tempfile.mkdtemp(prefix='noclear-uploads-'))
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


@dataclass
class Job:
    id: str
    filename: str
    status: str = 'queued'  # queued | running | done | error
    page: int = 0
    page_count: int | None = None
    rows: list[list] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        data = {
            'id': self.id,
            'filename': self.filename,
            'status': self.status,
            'page': self.page,
            'page_count': self.page_count,
            'error': self.error,
        }
        if self.status == 'done':
            data['header'] = CSV_HEADER
            data['rows'] = self.rows
        return data


jobs: dict[str, Job] = {}
# The OCR pipeline handles one PDF at a time, so jobs queue on a single worker thread.
executor = ThreadPoolExecutor(max_workers=1)
_pipeline = None


def get_pipeline():
    # Only ever called on the single worker thread, so no lock is needed.
    global _pipeline
    if _pipeline is None:
        _pipeline = create_pipeline()
    return _pipeline


def warm_up() -> None:
    try:
        get_pipeline()
    except Exception as e:  # e.g. vLLM not up yet; the first job will retry
        print(f'Pipeline not ready yet: {e}')


def run_job(job: Job, pdf_path: Path) -> None:
    job.status = 'running'

    def on_page(page_number: int, page_count: int | None) -> None:
        job.page = page_number
        job.page_count = page_count

    try:
        job.rows = extract_rows(pdf_path, get_pipeline(), on_page)
        write_csv(job.rows, OUTPUT_DIR / f'{Path(job.filename).stem}.csv')
        job.status = 'done'
    except Exception as e:
        job.error = str(e)
        job.status = 'error'
    finally:
        pdf_path.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    executor.submit(warm_up)  # load the models now so the first upload isn't slow
    yield
    executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title='noclear', lifespan=lifespan)


@app.get('/', response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML.read_text(encoding='utf-8')


@app.post('/api/jobs')
async def create_job(file: UploadFile) -> dict:
    filename = Path(file.filename or 'upload.pdf').name
    if not filename.lower().endswith('.pdf'):
        raise HTTPException(400, 'Only PDF files are supported.')

    job = Job(id=uuid.uuid4().hex, filename=filename)
    pdf_path = UPLOAD_DIR / f'{job.id}.pdf'
    size = 0
    with pdf_path.open('wb') as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                f.close()
                pdf_path.unlink(missing_ok=True)
                raise HTTPException(413, 'PDF is larger than 100 MB.')
            f.write(chunk)
    with pdf_path.open('rb') as f:
        if f.read(5) != b'%PDF-':
            pdf_path.unlink(missing_ok=True)
            raise HTTPException(400, 'That file is not a valid PDF.')

    jobs[job.id] = job
    executor.submit(run_job, job, pdf_path)
    return job.to_dict()


def _get_job(job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, 'Job not found.')
    return job


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str) -> dict:
    return _get_job(job_id).to_dict()


@app.get('/api/jobs/{job_id}/csv')
def download_csv(job_id: str) -> Response:
    job = _get_job(job_id)
    if job.status != 'done':
        raise HTTPException(409, 'This job has not finished yet.')
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(CSV_HEADER)
    writer.writerows(job.rows)
    csv_name = f'{Path(job.filename).stem}.csv'
    return Response(
        buffer.getvalue(),
        media_type='text/csv; charset=utf-8',
        headers={'Content-Disposition': f"attachment; filename*=UTF-8''{quote(csv_name)}"},
    )


def serve() -> None:
    import uvicorn

    host = os.environ.get('NOCLEAR_WEB_HOST', '127.0.0.1')
    port = int(os.environ.get('NOCLEAR_WEB_PORT', '8000'))
    uvicorn.run(app, host=host, port=port)
