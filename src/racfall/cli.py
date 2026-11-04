from __future__ import annotations

import argparse
import json

from .env import print_diagnostic
from .evaluation.events import evaluate, read_alerts, read_events, read_manifest


def main():
    parser = argparse.ArgumentParser(description="RAC-Fall research tools (not a safety product)")
    command = parser.add_subparsers(dest="command", required=True)
    command.add_parser("env-check", help="Check local Python / Apple MPS support")
    ev = command.add_parser("eval-events", help="Reference event-level evaluation for annotated CSVs")
    ev.add_argument("--manifest", required=True)
    ev.add_argument("--ground-truth", required=True)
    ev.add_argument("--alerts", required=True)
    ev.add_argument("--cooldown-s", type=float, default=10.0)
    ev.add_argument("--match-window-s", type=float, default=5.0)
    smoke = command.add_parser("pose-smoke", help="Run pretrained pose inference only (no fall decision)")
    smoke.add_argument("--source", required=True, help="MP4 path or webcam index 0")
    smoke.add_argument("--max-frames", type=int, default=50)
    smoke.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    smoke.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if args.command == "env-check":
        print_diagnostic()
    elif args.command == "eval-events":
        manifest = read_manifest(args.manifest)
        events = read_events(args.ground_truth, manifest)
        alerts = read_alerts(args.alerts, manifest)
        print(json.dumps(evaluate(manifest, events, alerts, cooldown_s=args.cooldown_s,
                                  match_window_s=args.match_window_s), indent=2))
    elif args.command == "pose-smoke":
        from .pose_smoke import pose_smoke
        print(json.dumps(pose_smoke(args.source, args.max_frames, args.device, args.imgsz), indent=2))


if __name__ == "__main__":
    main()
