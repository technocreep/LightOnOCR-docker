#!/usr/bin/env bash
set -euo pipefail

IMG_PATH="${1:-}"
URL="${OCR_URL:-http://localhost:8088}"
KEY="${OCR_API_KEY:-change-me}"

if [[ -z "$IMG_PATH" || "$IMG_PATH" == "-h" || "$IMG_PATH" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./scripts/test_image.sh /path/to/image.png

Env:
  OCR_URL      Base URL (default: http://localhost:8088)
  OCR_API_KEY  Bearer key (default: change-me)

Notes:
  - Sends PUT /process with raw bytes.
  - Prints first 1000 bytes of JSON response.
EOF
  exit 0
fi

if [[ ! -f "$IMG_PATH" ]]; then
  echo "File not found: $IMG_PATH" >&2
  exit 1
fi

FILENAME="$(basename "$IMG_PATH")"
MIME="$(file --brief --mime-type "$IMG_PATH" 2>/dev/null || true)"
if [[ -z "$MIME" ]]; then
  MIME="application/octet-stream"
fi

curl -sS -X PUT "$URL/process" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: $MIME" \
  -H "X-Filename: $FILENAME" \
  --data-binary "@$IMG_PATH" \
  | head -c 1000

echo
