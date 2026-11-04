"""Reference event-level evaluator: causal alerts and explicitly negative exposure.

This is *not* an algorithm for identifying falls. Its policy must be frozen
before test-set evaluation and consistently applied to all candidate models.
"""
from __future__ import annotations

import csv
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Video:
    video_id: str
    dataset: str
    subject_id: str
    split: str
    activity: str
    duration_s: float
    has_fall: bool


@dataclass(frozen=True)
class FallEvent:
    video_id: str
    event_id: str
    onset_s: float
    end_s: float


@dataclass(frozen=True)
class Alert:
    video_id: str
    alert_s: float


def _rows(path: str | Path, required: set[str]):
    with open(path, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path}: required CSV columns: {', '.join(sorted(required))}")
        for n, row in enumerate(reader, start=2):
            if any(row.get(k) is None or not str(row[k]).strip() for k in required):
                raise ValueError(f"{path}:{n}: missing required value")
            yield row


def _float(value: str, label: str) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected numeric seconds, got {value!r}") from exc
    if not math.isfinite(n):
        raise ValueError(f"{label}: seconds must be finite")
    return n


def read_manifest(path: str | Path) -> dict[str, Video]:
    fields = {"video_id", "dataset", "subject_id", "split", "activity", "duration_s", "has_fall"}
    videos: dict[str, Video] = {}
    subject_split: dict[tuple[str, str], str] = {}
    for row in _rows(path, fields):
        duration = _float(row["duration_s"], "duration_s")
        if duration <= 0:
            raise ValueError("duration_s must be positive")
        if row["has_fall"] not in ("0", "1"):
            raise ValueError("has_fall must be exactly 0 or 1")
        if row["video_id"] in videos:
            raise ValueError(f"Duplicate video_id {row['video_id']}")
        if row["split"] not in {"train", "val", "test", "external_dev", "blind_test"}:
            raise ValueError(f"Invalid split {row['split']!r}")
        vid = Video(row["video_id"], row["dataset"], row["subject_id"], row["split"],
                    row["activity"], duration, row["has_fall"] == "1")
        key = (vid.dataset, vid.subject_id)
        if key in subject_split and subject_split[key] != vid.split:
            raise ValueError(f"Subject leakage: {key} in {subject_split[key]} and {vid.split}")
        subject_split[key] = vid.split
        videos[vid.video_id] = vid
    if not videos:
        raise ValueError("Empty manifest")
    return videos


def read_events(path: str | Path, videos: dict[str, Video]) -> list[FallEvent]:
    result: list[FallEvent] = []
    seen: set[tuple[str, str]] = set()
    for row in _rows(path, {"video_id", "event_id", "onset_s", "end_s"}):
        vid = row["video_id"]
        if vid not in videos:
            raise ValueError(f"Unknown ground-truth video {vid}")
        if not videos[vid].has_fall:
            raise ValueError(f"Event in has_fall=0 clip {vid}")
        start = _float(row["onset_s"], "onset_s")
        end = _float(row["end_s"], "end_s")
        if start < 0 or end < start or end > videos[vid].duration_s:
            raise ValueError(f"Invalid event time bounds for {vid}")
        key = (vid, row["event_id"])
        if key in seen:
            raise ValueError(f"Duplicate event key {key}")
        seen.add(key)
        result.append(FallEvent(vid, row["event_id"], start, end))
    missing = {v.video_id for v in videos.values() if v.has_fall} - {e.video_id for e in result}
    if missing:
        raise ValueError(f"Missing onset annotations for has_fall=1 videos: {sorted(missing)}")
    return result


def read_alerts(path: str | Path, videos: dict[str, Video]) -> list[Alert]:
    result = []
    for row in _rows(path, {"video_id", "alert_s"}):
        vid = row["video_id"]
        if vid not in videos:
            raise ValueError(f"Unknown predicted-alert video {vid}")
        t = _float(row["alert_s"], "alert_s")
        if t < 0 or t > videos[vid].duration_s:
            raise ValueError(f"Invalid alert time {t} in {vid}")
        result.append(Alert(vid, t))
    return result


def evaluate(
    videos: dict[str, Video], events: list[FallEvent], alerts: list[Alert],
    *, cooldown_s: float = 10.0, match_window_s: float = 5.0
) -> dict:
    """Greedy one-to-one matching; report false alarms/hour on pure ADL footage.

    For each video, suppress alert timestamps closer than `cooldown_s` to the
    previous *retained* alert. Such policy is fixed for all model comparisons.
    """
    if not math.isfinite(cooldown_s) or cooldown_s < 0:
        raise ValueError("cooldown_s must be finite and >= 0")
    if not math.isfinite(match_window_s) or match_window_s < 0:
        raise ValueError("match_window_s must be finite and >= 0")
    # Validate direct callers as well as CSV callers.
    for event in events:
        if event.video_id not in videos or not videos[event.video_id].has_fall:
            raise ValueError(f"Invalid event video {event.video_id}")
    for alert in alerts:
        if alert.video_id not in videos:
            raise ValueError(f"Invalid alert video {alert.video_id}")
    by_vid: dict[str, list[Alert]] = {vid: [] for vid in videos}
    for a in alerts:
        by_vid[a.video_id].append(a)
    retained: dict[str, list[Alert]] = {}
    for vid, times in by_vid.items():
        kept: list[Alert] = []
        for a in sorted(times, key=lambda z: z.alert_s):
            if not kept or a.alert_s - kept[-1].alert_s >= cooldown_s:
                kept.append(a)
        retained[vid] = kept

    used: set[tuple[str, int]] = set()
    delays: list[float] = []
    matched_events: list[str] = []
    for event in sorted(events, key=lambda e: (e.video_id, e.onset_s, e.event_id)):
        available = retained.get(event.video_id, [])
        for i, alert in enumerate(available):
            if (event.video_id, i) in used:
                continue
            if event.onset_s <= alert.alert_s <= event.onset_s + match_window_s:
                used.add((event.video_id, i))
                delays.append(alert.alert_s - event.onset_s)
                matched_events.append(f"{event.video_id}:{event.event_id}")
                break

    tp = len(matched_events)
    fn = len(events) - tp
    retained_count = sum(map(len, retained.values()))
    fp = retained_count - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    negative_duration_s = sum(v.duration_s for v in videos.values() if not v.has_fall)
    adl_alerts = sum(len(retained[vid]) for vid, v in videos.items() if not v.has_fall)
    return {
        "tp": tp, "fp_all_unmatched": fp, "fn": fn,
        "event_precision": precision, "event_recall": recall, "event_f1": f1,
        "median_detection_delay_s": statistics.median(delays) if delays else None,
        "adl_false_alerts": adl_alerts,
        "adl_exposure_hours": negative_duration_s / 3600,
        "false_alarms_per_adl_hour": adl_alerts / (negative_duration_s / 3600) if negative_duration_s > 0 else None,
        "retained_alerts": retained_count,
        "total_ground_truth_events": len(events),
        "policy": {"cooldown_s": cooldown_s, "match_window_s": match_window_s, "matching": "causal_onset_to_onset_plus_window"},
    }
