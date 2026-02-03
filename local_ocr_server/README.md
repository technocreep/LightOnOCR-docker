# Local OCR Server (GPU, Docker)

Minimal OCR HTTP API compatible with OpenWebUI's `external` document loader.

It implements:
- `PUT /process` (OpenWebUI calls this)
- `GET /healthz`

## Requirements

- Docker Engine
- NVIDIA GPU + drivers
- NVIDIA Container Toolkit (so Docker can use `--gpus all`)

## Quick start (Docker Compose)
From this folder:

```bash
docker compose up -d --build
```

Server listens on `http://localhost:8088`.

Health check:

```bash
curl -s http://localhost:8088/healthz | jq
```

Real-time logs:

```bash
docker compose logs -f local-ocr
```

## Test scripts

Image:

```bash
python3 scripts/test_image.py /path/to/image.png
```

PDF:

```bash
python3 scripts/test_pdf.py /path/to/file.pdf
```

## OpenWebUI configuration

This server does **not** enforce API keys.

Note: OpenWebUI still requires a **non-empty** key value to enable some engines.
You can set any string (example: `local`).

### Option A: OpenWebUI `external` loader

- `CONTENT_EXTRACTION_ENGINE=external`
- `EXTERNAL_DOCUMENT_LOADER_URL=http://local-ocr:8080` (if same compose network)
- `EXTERNAL_DOCUMENT_LOADER_API_KEY=local` (any non-empty string)

### Option B: OpenWebUI `mistral_ocr` loader

- `CONTENT_EXTRACTION_ENGINE=mistral_ocr`
- `MISTRAL_OCR_API_BASE_URL=http://local-ocr:8080/v1`
- `MISTRAL_OCR_API_KEY=local` (any non-empty string)

If OpenWebUI is *not* on the same Docker network, use your host IP, for example:

- `EXTERNAL_DOCUMENT_LOADER_URL=http://<your-host-ip>:8088`

## API

### `PUT /process`

OpenWebUI sends the raw file bytes in the request body.

Expected headers:
- `Content-Type: application/pdf` (or image mime)
- `X-Filename: <name>`

Returns either:
- a JSON **list** of `{ page_content, metadata }` (PDF, per page)
- or a JSON **object** `{ page_content, metadata }` (single image)

## Tuning

Environment variables:
- `OCR_API_KEY` (recommended in production)
- `OCR_LANGS` (comma-separated, default `en`)
- `OCR_GPU` = `auto` (default), `true`, `false`
- `PDF_RENDER_DPI` (default `200`)
- `PDF_MIN_TEXT_CHARS` (default `40`) — if PDF already contains embedded text, OCR is skipped

## Notes

- For PDFs with embedded text, the server returns extracted text (fast).
- For scanned PDFs, it renders pages to images and runs EasyOCR (uses GPU if available).

## Troubleshooting

### GPU not visible

Test:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

If that fails, install/configure NVIDIA Container Toolkit.
