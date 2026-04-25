"""Fetch audio from a URL (YouTube, direct media link, etc.) into a local wav."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

MAX_DURATION_S = int(os.environ.get("INGEST_MAX_DURATION_S", "1800"))  # 30 min default; override via env


class IngestionError(Exception):
    pass


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def fetch_audio(url: str, dest_dir: Path) -> Path:
    if not re.match(r"^https?://", url):
        raise IngestionError("URL must start with http(s)://")
    if not _have("yt-dlp"):
        raise IngestionError("yt-dlp not installed; run: pip install yt-dlp")
    if not _have("ffmpeg"):
        raise IngestionError("ffmpeg not found in PATH; install via brew/apt")

    dest_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(dest_dir / f"{uuid.uuid4().hex}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--no-playlist",
        "--match-filter", f"duration <= {MAX_DURATION_S}",
        "-o", out_template,
        url,
    ]
    log.info("ingesting: %s", url)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "yt-dlp failed").strip().splitlines()[-1]
        raise IngestionError(msg[:300])

    base = Path(out_template).with_suffix("")
    wav = base.with_suffix(".wav")
    if not wav.exists():
        # Try to find whatever yt-dlp produced
        candidates = list(dest_dir.glob(f"{base.name}.*"))
        if not candidates:
            raise IngestionError("no audio produced")
        return candidates[0]
    return wav
