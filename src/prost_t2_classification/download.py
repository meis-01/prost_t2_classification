from __future__ import annotations

import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Sequence, Set
from urllib.parse import parse_qs, urlsplit, urlunsplit

from .logging_utils import get_logger


_CURL_RE = re.compile(r'curl\s+-C\s+-\s+"(?P<url>[^"]+)"\s+--output\s+(?P<output>\S+)')
_ARCHIVE_SUFFIXES = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
)


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
        validate_download_url(entry)
        output_path = download_dir / entry.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
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
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"curl failed with exit code {exc.returncode} while downloading {entry.output} "
                f"from {redact_url(entry.url)}. Check your network connection and regenerate the "
                "fastMRI download script if the link has expired."
            ) from exc

    return paths


def extract_archives(
    archive_paths: Iterable[Path],
    extract_dir: Path,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    source_root: Path | None = None,
) -> None:
    logger = get_logger()
    extract_dir.mkdir(parents=True, exist_ok=True)
    resolved_source_root = source_root.resolve() if source_root is not None else None
    for archive_path in archive_paths:
        if not is_archive_path(archive_path):
            relative_path = relative_to_source_root(archive_path, resolved_source_root)
            output_path = extract_dir / relative_path
            if output_path.exists() and not overwrite:
                logger.info("Skipping existing raw file %s", output_path)
                continue
            logger.info("Staging raw file %s into %s", archive_path, output_path)
            if dry_run:
                continue
            if not archive_path.exists():
                raise FileNotFoundError(f"Raw file does not exist: {archive_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive_path, output_path)
            continue

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


def is_archive_path(path: Path | str) -> bool:
    lowered = str(path).lower()
    return any(lowered.endswith(suffix) for suffix in _ARCHIVE_SUFFIXES)


def relative_to_source_root(path: Path, source_root: Path | None) -> Path:
    if source_root is None:
        return Path(path.name)
    resolved_path = path.resolve()
    if resolved_path.is_relative_to(source_root):
        return resolved_path.relative_to(source_root)
    return Path(path.name)


def select_entries_for_rawfiles(
    entries: Sequence[DownloadEntry],
    rawfiles: Iterable[str],
) -> List[DownloadEntry]:
    rawfile_names: Set[str] = {Path(rawfile).name for rawfile in rawfiles}
    rawfile_stems: Set[str] = {Path(rawfile).stem for rawfile in rawfile_names}
    return [
        entry
        for entry in entries
        if any(candidate in rawfile_names or Path(candidate).stem in rawfile_stems for candidate in entry_filenames(entry))
    ]


def entry_filenames(entry: DownloadEntry) -> Set[str]:
    url_path = urlsplit(entry.url).path
    return {
        Path(entry.output).name,
        PurePosixPath(entry.output).name,
        PurePosixPath(url_path).name,
    }


def validate_download_url(entry: DownloadEntry) -> None:
    expiration = presigned_url_expiration(entry.url)
    if expiration is None:
        return

    now = datetime.now(timezone.utc)
    if expiration <= now:
        raise ValueError(
            f"The download URL for {entry.output} expired on {format_utc(expiration)}. "
            "fastMRI download links are time-limited; request a fresh prostate download script "
            "from fastMRI and rerun the command."
        )


def presigned_url_expiration(url: str) -> datetime | None:
    query = parse_qs(urlsplit(url).query)
    expires = first_query_value(query, "Expires")
    if expires is not None:
        try:
            return datetime.fromtimestamp(int(expires), timezone.utc)
        except ValueError:
            return None

    amz_date = first_query_value(query, "X-Amz-Date")
    amz_expires = first_query_value(query, "X-Amz-Expires")
    if amz_date is None or amz_expires is None:
        return None
    try:
        issued_at = datetime.strptime(amz_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return issued_at + timedelta(seconds=int(amz_expires))
    except ValueError:
        return None


def first_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    return values[0]


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def redact_url(url: str) -> str:
    split = urlsplit(url)
    return urlunsplit((split.scheme, split.netloc, split.path, "", ""))
