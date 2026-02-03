#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request
import urllib.parse
from pathlib import Path


def _default_url() -> str:
    return os.environ.get("OCR_URL", "http://localhost:8088").rstrip("/")


def _default_key() -> str:
    return os.environ.get("OCR_API_KEY", "change-me")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Test Local OCR Server with a PDF via PUT /process (stdlib only)."
    )
    p.add_argument("pdf", type=Path, help="Path to PDF")
    p.add_argument("--url", default=_default_url(), help="Base URL (env OCR_URL)")
    p.add_argument("--key", default=_default_key(), help="API key (env OCR_API_KEY)")
    p.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("OCR_TIMEOUT", "300")),
        help="Request timeout seconds (env OCR_TIMEOUT, default 300)",
    )
    p.add_argument(
        "--max-bytes",
        type=int,
        default=2000,
        help="Print only first N bytes of response body",
    )
    args = p.parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.is_file():
        print(f"File not found: {pdf_path}", file=sys.stderr)
        return 1

    data = pdf_path.read_bytes()

    req = urllib.request.Request(
        url=f"{args.url.rstrip('/')}/process",
        data=data,
        method="PUT",
        headers={
            "Authorization": f"Bearer {args.key}",
            "Content-Type": "application/pdf",
            "X-Filename": urllib.parse.quote(pdf_path.name),
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            # Print selected response headers
            print("--- Response headers (selected) ---")
            for k, v in resp.headers.items():
                if k.lower().startswith("x-ocr"):
                    print(f"{k}: {v}")

            body = resp.read()
            print("--- Response body (first bytes) ---")
            print(body[: args.max_bytes].decode("utf-8", errors="replace"))
            if len(body) > args.max_bytes:
                print(f"\n... (truncated, {len(body)} bytes total)")

    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
