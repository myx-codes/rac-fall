"""Controlled official-file download for canonical CAUCAFall PNG/TXT data."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OFFICIAL_PREFIX = "https://data.mendeley.com/public-files/datasets/7w7fccy7ky/files/"
ALLOWED_SUBJECTS = tuple(f"Subject.{number}" for number in range(1, 9))


@dataclass(frozen=True)
class OfficialFile:
    subject: str
    action: str
    filename: str
    file_id: str
    size: int
    sha256: str
    download_url: str

    @property
    def relative_path(self) -> Path:
        return Path(self.subject) / self.action / self.filename


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_inventory(
    folders_json: str | Path,
    metadata_dir: str | Path,
    *,
    subjects: tuple[str, ...] = ALLOWED_SUBJECTS,
) -> list[OfficialFile]:
    """Load an exact PNG/TXT inventory without consulting sealed subject listings."""
    requested = tuple(subjects)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("subjects must be a non-empty unique list")
    if not set(requested) <= set(ALLOWED_SUBJECTS):
        raise ValueError("Only CAUCAFall Subjects 1-8 may be downloaded in Phase 2B.5")
    folder_rows = json.loads(Path(folders_json).read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in folder_rows}
    subject_by_id = {
        row["id"]: row["name"] for row in folder_rows if row["name"] in requested
    }
    action_folders: dict[str, tuple[str, str]] = {}
    for row in folder_rows:
        parent = row.get("parent_id")
        if parent in subject_by_id:
            action_folders[row["id"]] = (subject_by_id[parent], row["name"])
    inventory: list[OfficialFile] = []
    seen: set[Path] = set()
    metadata_root = Path(metadata_dir)
    for folder_id, (subject, action) in sorted(action_folders.items()):
        matches = sorted(metadata_root.glob(f"{subject}_{folder_id}.json"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one official metadata listing for {subject}/{action} ({folder_id}), "
                f"found {len(matches)}"
            )
        rows = json.loads(matches[0].read_text(encoding="utf-8"))
        for row in rows:
            filename = str(row.get("filename", ""))
            suffix = Path(filename).suffix.lower()
            if suffix not in {".png", ".txt"}:
                continue
            details = row.get("content_details") or {}
            file_id = str(row.get("id", ""))
            size = int(details.get("size", row.get("size", -1)))
            sha256 = str(details.get("sha256_hash", "")).lower()
            url = str(details.get("download_url", ""))
            expected_url = f"{OFFICIAL_PREFIX}{file_id}/file_downloaded"
            if (not file_id or size < 0 or len(sha256) != 64 or
                    any(c not in "0123456789abcdef" for c in sha256) or
                    url != expected_url):
                raise ValueError(f"Invalid official metadata for {subject}/{action}/{filename}")
            item = OfficialFile(subject, action, filename, file_id, size, sha256, url)
            if item.relative_path in seen:
                raise ValueError(f"Duplicate official path: {item.relative_path}")
            seen.add(item.relative_path)
            inventory.append(item)
    missing_subjects = sorted(set(requested) - {item.subject for item in inventory})
    if missing_subjects:
        raise ValueError(f"No official PNG/TXT inventory for subjects: {missing_subjects}")
    return sorted(inventory, key=lambda item: (int(item.subject.split(".")[1]), item.action,
                                                item.filename))


def _verified(path: Path, item: OfficialFile) -> bool:
    return path.is_file() and path.stat().st_size == item.size and sha256_file(path) == item.sha256


def _build_reuse_index(root: Path, destination: Path) -> dict[tuple[int, str], Path]:
    index: dict[tuple[int, str], Path] = {}
    if not root.is_dir():
        return index
    for path in root.rglob("*"):
        if (not path.is_file() or path.suffix.lower() not in {".png", ".txt"} or
                path.name.endswith(".partial")):
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        index.setdefault((size, digest), path)
    return index


def plan_download(
    inventory: list[OfficialFile], destination: str | Path, reuse_root: str | Path,
) -> dict:
    destination_path = Path(destination).resolve()
    reuse = _build_reuse_index(Path(reuse_root).resolve(), destination_path)
    counts = {"verified": 0, "reusable": 0, "network": 0}
    bytes_by_status = {key: 0 for key in counts}
    for item in inventory:
        target = destination_path / item.relative_path
        if target.exists():
            if not _verified(target, item):
                raise ValueError(f"Existing target fails official size/SHA-256: {target}")
            status = "verified"
        elif (item.size, item.sha256) in reuse:
            status = "reusable"
        else:
            status = "network"
        counts[status] += 1
        bytes_by_status[status] += item.size
    return {
        "official_files": len(inventory),
        "official_bytes": sum(item.size for item in inventory),
        "counts": counts,
        "bytes": bytes_by_status,
    }


def _download_one(item: OfficialFile, target: Path, *, retries: int, timeout_s: float) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    for attempt in range(retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item.size:
            partial.unlink()
            offset = 0
        headers = {"User-Agent": "RAC-Fall/0.1 official dataset downloader"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(item.download_url, headers=headers), timeout=timeout_s) as response:
                status = getattr(response, "status", response.getcode())
                mode = "ab" if offset and status == 206 else "wb"
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            if _verified(partial, item):
                os.replace(partial, target)
                return item.size
            error = ValueError(
                f"Downloaded size/SHA-256 mismatch for {item.relative_path}: "
                f"got {partial.stat().st_size if partial.exists() else 0} bytes"
            )
            partial.unlink(missing_ok=True)
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise PermissionError(
                    f"Official Mendeley download denied with HTTP {exc.code}: {item.download_url}"
                ) from exc
            error = exc
        except (URLError, TimeoutError, OSError) as exc:
            error = exc
        if attempt == retries:
            raise RuntimeError(
                f"Download failed after {retries + 1} attempts for {item.relative_path}: {error}"
            ) from error
        time.sleep(min(2 ** attempt, 8))
    raise AssertionError("unreachable")


def download_official_canonical(
    inventory: list[OfficialFile],
    destination: str | Path,
    reuse_root: str | Path,
    report_dir: str | Path,
    *,
    workers: int = 8,
    retries: int = 3,
    timeout_s: float = 120.0,
    min_free_bytes: int = 2 * 1024 ** 3,
) -> dict:
    if workers <= 0 or retries < 0 or timeout_s <= 0 or min_free_bytes < 0:
        raise ValueError("Invalid downloader limits")
    destination_path = Path(destination).resolve()
    report_path = Path(report_dir).resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)
    reuse = _build_reuse_index(Path(reuse_root).resolve(), destination_path)
    initial_plan = plan_download(inventory, destination_path, reuse_root)
    required = initial_plan["bytes"]["network"] + initial_plan["bytes"]["reusable"]
    disk = shutil.disk_usage(destination_path)
    if disk.free < required + min_free_bytes:
        raise OSError(
            f"Insufficient disk: free={disk.free}, required_growth={required}, "
            f"reserve={min_free_bytes}"
        )
    started = time.perf_counter()
    summaries = []
    lock = threading.Lock()
    for subject in ALLOWED_SUBJECTS:
        subject_items = [item for item in inventory if item.subject == subject]
        if not subject_items:
            continue
        subject_started = time.perf_counter()
        counters = {"verified": 0, "reused": 0, "downloaded": 0, "bytes_downloaded": 0}
        pending: list[tuple[OfficialFile, Path]] = []
        for item in subject_items:
            target = destination_path / item.relative_path
            if target.exists():
                if not _verified(target, item):
                    raise ValueError(f"Existing target fails official size/SHA-256: {target}")
                counters["verified"] += 1
                continue
            source = reuse.get((item.size, item.sha256))
            if source:
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".partial")
                shutil.copyfile(source, partial)
                if not _verified(partial, item):
                    raise ValueError(f"Reusable source changed or failed checksum: {source}")
                os.replace(partial, target)
                counters["reused"] += 1
            else:
                pending.append((item, target))
        if shutil.disk_usage(destination_path).free < (
            sum(item.size for item, _ in pending) + min_free_bytes
        ):
            raise OSError(f"Insufficient disk before downloading {subject}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_download_one, item, target, retries=retries, timeout_s=timeout_s): item
                for item, target in pending
            }
            for future in as_completed(futures):
                size = future.result()
                with lock:
                    counters["downloaded"] += 1
                    counters["bytes_downloaded"] += size
        for item in subject_items:
            target = destination_path / item.relative_path
            if not _verified(target, item):
                raise ValueError(f"Post-download verification failed: {target}")
        subject_summary = {
            "subject": subject,
            "official_files": len(subject_items),
            "official_bytes": sum(item.size for item in subject_items),
            **counters,
            "elapsed_s": round(time.perf_counter() - subject_started, 6),
            "free_bytes_after": shutil.disk_usage(destination_path).free,
        }
        summaries.append(subject_summary)
        _atomic_json(report_path / f"download-{subject}.json", subject_summary)
    result = {
        "schema_version": "racfall.caucafall-download.v1",
        "source": "official Mendeley CAUCAFall v5 public-files endpoints only",
        "destination": str(destination_path),
        "initial_plan": initial_plan,
        "subjects": summaries,
        "elapsed_s": round(time.perf_counter() - started, 6),
        "free_bytes_after": shutil.disk_usage(destination_path).free,
    }
    _atomic_json(report_path / "download-summary.json", result)
    return result
