#!/usr/bin/env python3
import argparse
import mimetypes
import os
import sys
import urllib.request
import urllib.parse
from pathlib import Path


def _default_url() -> str:
    return os.environ.get("OCR_URL", "http://localhost:8088").rstrip("/")


def _default_key() -> str:
    return os.environ.get("OCR_API_KEY", "change-me")


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Test Local OCR Server with an image via PUT /process (stdlib only)."
    )
    p.add_argument("image", type=Path, help="Path to image (png/jpg/webp/etc)")
    p.add_argument("--url", default=_default_url(), help="Base URL (env OCR_URL)")
    p.add_argument("--key", default=_default_key(), help="API key (env OCR_API_KEY)")
    p.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("OCR_TIMEOUT", "120")),
        help="Request timeout seconds (env OCR_TIMEOUT, default 120)",
    )
    p.add_argument(
        "--max-bytes",
        type=int,
        default=1000,
        help="Print only first N bytes of response body",
    )
    args = p.parse_args()

    img_path: Path = args.image
    if not img_path.is_file():
        print(f"File not found: {img_path}", file=sys.stderr)
        return 1

    data = img_path.read_bytes()
    mime = _guess_mime(img_path)

    req = urllib.request.Request(
        url=f"{args.url.rstrip('/')}/process",
        data=data,
        method="PUT",
        headers={
            "Authorization": f"Bearer {args.key}",
            "Content-Type": mime,
            "X-Filename": urllib.parse.quote(img_path.name),
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            body = resp.read()
            print(body[: args.max_bytes].decode("utf-8", errors="replace"))
            if len(body) > args.max_bytes:
                print(f"\n... (truncated, {len(body)} bytes total)")
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
