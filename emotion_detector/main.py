"""Command line entry point for the real-time emotion detector."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2

from .analyzer import EmotionAnalyzer, EmotionDetection, format_detection_summary
from .config import EmotionWeights, TrackerConfig, merge_emotion_weights
from .engagement import EngagementTracker, FrameSummary
from .visualizer import draw_detections


def _comma_separated(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time facial emotion and engagement analysis for video calls.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: camera index (default 0) or path/URL to a video stream.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the OpenCV preview window (useful for headless environments).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Optional CSV file to store per-frame engagement metrics.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=5.0,
        help="How often (in seconds) to print aggregated engagement statistics.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional limit on the number of frames to process (useful for testing).",
    )
    parser.add_argument(
        "--disable-mtcnn",
        action="store_true",
        help="Disable the MTCNN face detector in FER (falls back to Haar cascades).",
    )
    parser.add_argument(
        "--positive-emotions",
        type=str,
        help="Comma separated list of emotions treated as highly engaged.",
    )
    parser.add_argument(
        "--neutral-emotions",
        type=str,
        help="Comma separated list of neutral emotions for engagement scoring.",
    )
    parser.add_argument(
        "--negative-emotions",
        type=str,
        help="Comma separated list of disengaged emotions.",
    )
    return parser


def _open_video_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
    else:
        capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
    return capture


def _prepare_tracker(args: argparse.Namespace) -> EngagementTracker:
    weights = EmotionWeights()
    overrides = {
        "positive": _comma_separated(args.positive_emotions),
        "neutral": _comma_separated(args.neutral_emotions),
        "negative": _comma_separated(args.negative_emotions),
    }
    overrides = {k: v for k, v in overrides.items() if v}
    if overrides:
        weights = merge_emotion_weights(weights, overrides)
    config = TrackerConfig(emotion_weights=weights)
    return EngagementTracker(config=config)


def _setup_logging(path: Optional[str]):
    if not path:
        return None, None

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handle = log_path.open("w", newline="")
    writer = csv.DictWriter(
        file_handle,
        fieldnames=[
            "timestamp",
            "raw_score",
            "smoothed_score",
            "rolling_average",
            "dominant_emotion",
            "emotion_distribution",
            "faces_detected",
        ],
    )
    writer.writeheader()
    return file_handle, writer


def _log_frame(writer, summary: FrameSummary, detections: List[EmotionDetection]):
    if not writer:
        return

    writer.writerow(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "raw_score": f"{summary.raw_score:.4f}",
            "smoothed_score": f"{summary.smoothed_score:.4f}",
            "rolling_average": f"{summary.rolling_average:.4f}",
            "dominant_emotion": summary.dominant_emotion or "",
            "emotion_distribution": ";".join(
                f"{emotion}:{score:.2f}" for emotion, score in summary.emotion_distribution.items()
            ),
            "faces_detected": len(detections),
        }
    )


def _print_summary(summary: FrameSummary, detections: List[EmotionDetection]):
    detection_summary = format_detection_summary(detections)
    dominant = summary.dominant_emotion or "n/a"
    print(
        "[Engagement] raw={:.2f} smoothed={:.2f} avg={:.2f} dominant={} emotions={}".format(
            summary.raw_score,
            summary.smoothed_score,
            summary.rolling_average,
            dominant,
            ", ".join(f"{emotion}:{score:.2f}" for emotion, score in detection_summary.items())
            or "n/a",
        ),
        flush=True,
    )


def run(args: Optional[List[str]] = None) -> None:
    parser = build_argument_parser()
    parsed_args = parser.parse_args(args=args)

    display = not parsed_args.no_display
    analyzer = EmotionAnalyzer(use_mtcnn=not parsed_args.disable_mtcnn)
    tracker = _prepare_tracker(parsed_args)

    capture = _open_video_source(parsed_args.source)
    log_handle, log_writer = _setup_logging(parsed_args.log_file)

    last_print_time = time.monotonic()
    frame_counter = 0

    try:
        while True:
            if parsed_args.max_frames is not None and frame_counter >= parsed_args.max_frames:
                break

            success, frame = capture.read()
            if not success or frame is None:
                print("Video stream ended or cannot be read.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = analyzer.detect(rgb_frame)
            summary = tracker.update(detections)

            _log_frame(log_writer, summary, detections)

            current_time = time.monotonic()
            if current_time - last_print_time >= parsed_args.print_interval:
                _print_summary(summary, detections)
                last_print_time = current_time

            if display:
                try:
                    annotated = draw_detections(frame, detections, summary)
                    cv2.imshow("Emotion Engagement Monitor", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                except cv2.error as error:
                    print(
                        f"OpenCV display failed: {error}. Continuing without on-screen preview.",
                        file=sys.stderr,
                    )
                    display = False

            frame_counter += 1
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()
        if log_handle:
            log_handle.close()


if __name__ == "__main__":
    run()
