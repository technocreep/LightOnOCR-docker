#!/usr/bin/env bash
set -euo pipefail

PDF_PATH="${1:-}"
URL="${OCR_URL:-http://localhost:8088}"
KEY="${OCR_API_KEY:-change-me}"

if [[ -z "$PDF_PATH" || "$PDF_PATH" == "-h" || "$PDF_PATH" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./scripts/test_pdf.sh /path/to/file.pdf

Env:
  OCR_URL      Base URL (default: http://localhost:8088)
  OCR_API_KEY  Bearer key (default: change-me)

Notes:
  - Sends PUT /process with raw PDF bytes.
  - Prints first 2000 bytes of JSON response (per-page docs).
EOF
  exit 0
fi

if [[ ! -f "$PDF_PATH" ]]; then
  echo "File not found: $PDF_PATH" >&2
  exit 1
fi

FILENAME="$(basename "$PDF_PATH")"

curl -sS -D /tmp/ocr_headers.txt -o /tmp/ocr_body.json -X PUT "$URL/process" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/pdf" \
  -H "X-Filename: $FILENAME" \
  --data-binary "@$PDF_PATH"

echo "--- Response headers (selected) ---"
grep -iE '^(HTTP/|x-ocr-)' /tmp/ocr_headers.txt || true

echo "--- Response body (first 2000 bytes) ---"
head -c 2000 /tmp/ocr_body.json

echo
