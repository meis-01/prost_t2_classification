from __future__ import annotations

import re
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
from urllib.parse import urlsplit, urlunsplit

from .logging_utils import get_logger


_CURL_RE = re.compile(r'curl\s+-C\s+-\s+"(?P<url>[^"]+)"\s+--output\s+(?P<output>\S+)')


@dataclass(frozen=True)
class DownloadEntry:
    url: str
    output: str
    kind: str


def parse_prostate_download_script(path: Path) -> List[DownloadEntry]:
    entries: List[DownloadEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _CURL_RE.search(line.strip())
        if not match:
            continue
        output = match.group("output")
        kind = classify_output(output)
        if kind in {"labels", "t2"}:
            entries.append(DownloadEntry(url=match.group("url"), output=output, kind=kind))
    if not entries:
        raise ValueError(f"No labels or T2 curl commands found in {path}.")
    return entries


def classify_output(output: str) -> str:
    lowered = output.lower()
    if lowered == "labels.tar.gz" or "labels" in lowered:
        return "labels"
    if "prostate_t2" in lowered or "axt2" in lowered:
        return "t2"
    return "other"


def download_entries(
    entries: Sequence[DownloadEntry],
    download_dir: Path,
    *,
    curl_executable: str = "curl",
    overwrite: bool = False,
    dry_run: bool = False,
) -> List[Path]:
    logger = get_logger()
    download_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    for entry in entries:
        output_path = download_dir / entry.output
        paths.append(output_path)
        if output_path.exists() and not overwrite:
            logger.info("Skipping existing archive %s", output_path)
            continue

        logger.info("Downloading %s archive %s from %s", entry.kind, entry.output, redact_url(entry.url))
        if dry_run:
            continue

        command = [
            curl_executable,
            "-L",
            "-C",
            "-",
            entry.url,
            "--output",
            str(output_path),
        ]
        subprocess.run(command, check=True)

    return paths


def extract_archives(
    archive_paths: Iterable[Path],
    extract_dir: Path,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    logger = get_logger()
    extract_dir.mkdir(parents=True, exist_ok=True)
    for archive_path in archive_paths:
        marker = extract_dir / f".{archive_path.name}.extracted"
        if marker.exists() and not overwrite:
            logger.info("Skipping previously extracted archive %s", archive_path)
            continue

        logger.info("Extracting %s into %s", archive_path, extract_dir)
        if dry_run:
            continue
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive does not exist: {archive_path}")

        with tarfile.open(archive_path, mode="r:*") as tar:
            safe_extract(tar, extract_dir)
        marker.write_text("ok\n", encoding="utf-8")


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if destination != target and destination not in target.parents:
            raise RuntimeError(f"Unsafe archive member path: {member.name}")
    tar.extractall(destination)


def redact_url(url: str) -> str:
    split = urlsplit(url)
    return urlunsplit((split.scheme, split.netloc, split.path, "", ""))
