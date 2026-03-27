#!/usr/bin/env python3
"""Drone tracking on video using a trained YOLO .pt model (Ultralytics Track mode)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_default_model() -> Path | None:
    """Find a default .pt model in project root.

    Priority:
    1) yolo26n*.pt
    2) best.pt
    3) first *.pt found in root
    """
    candidates = sorted(PROJECT_ROOT.glob("*.pt"))
    if not candidates:
        return None

    for model in candidates:
        if model.stem.startswith("yolo26n"):
            return model

    for model in candidates:
        if model.name == "best.pt":
            return model

    return candidates[0]


def parse_args(default_model: Path | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track drones in a video with a custom Ultralytics YOLO model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model,
        help=(
            "Path to trained .pt model. "
            "If omitted, auto-loads a .pt file from project root when available."
        ),
    )
    parser.add_argument("--source", type=Path, required=True, help="Path to input video")
    parser.add_argument(
        "--tracker",
        type=str,
        default="botsort.yaml",
        help="Tracker config: botsort.yaml / bytetrack.yaml or custom yaml path",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Class IDs to track, e.g. --classes 0",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display real-time tracking window (recommended on local macOS)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated output video under runs/track",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device, e.g. cpu, mps, cuda:0 (default: ultralytics auto)",
    )
    return parser.parse_args()


def validate_inputs(model_path: Path | None, source_path: Path) -> Path:
    if model_path is None:
        raise FileNotFoundError(
            "No model was provided and no .pt file was found in project root. "
            "Please place your model in root or pass --model /path/to/model.pt"
        )

    if not model_path.exists() or model_path.suffix.lower() != ".pt":
        raise FileNotFoundError(f"Invalid model path (expect .pt): {model_path}")

    if not source_path.exists() or source_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise FileNotFoundError(f"Invalid source video path: {source_path}")

    return model_path


def main() -> int:
    default_model = find_default_model()
    args = parse_args(default_model)

    try:
        model_path = validate_inputs(args.model, args.source)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] Using model: {model_path}")
    model = YOLO(str(model_path))

    # Keep stream=True to process long video robustly and persist=True to preserve IDs across frames.
    results = model.track(
        source=str(args.source),
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        classes=args.classes,
        show=args.show,
        save=args.save_video,
        stream=True,
        persist=True,
        device=args.device,
        verbose=True,
    )

    frame_count = 0
    unique_track_ids: set[int] = set()

    for result in results:
        frame_count += 1
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            unique_track_ids.update(ids)

        # Keep OpenCV UI responsive in --show mode on macOS.
        if args.show and (cv2.waitKey(1) & 0xFF == ord("q")):
            print("[INFO] Received 'q', exiting early.")
            break

    if args.show:
        cv2.destroyAllWindows()

    print("\n[SUMMARY]")
    print(f"Frames processed: {frame_count}")
    print(f"Unique tracked objects: {len(unique_track_ids)}")
    if args.save_video:
        print("Annotated video is saved under runs/track/* by Ultralytics.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
