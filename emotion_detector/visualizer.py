"""Rendering helpers for the emotion detector."""

from __future__ import annotations

from typing import Iterable, Optional

import cv2

from .analyzer import EmotionDetection
from .engagement import FrameSummary

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _format_percentage(value: float) -> str:
    return f"{value * 100:.0f}%"


def draw_detections(
    frame,
    detections: Iterable[EmotionDetection],
    summary: Optional[FrameSummary] = None,
):
    """Draw bounding boxes and emotion labels on a frame."""

    overlay = frame.copy()
    for detection in detections:
        x, y, w, h = detection.box
        dominant = detection.dominant_emotion or "unknown"
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 2)
        label = f"{dominant}: {_format_percentage(detection.emotions.get(dominant, 0.0))}"
        cv2.putText(overlay, label, (x, y - 10), FONT, 0.6, (255, 255, 0), 2)

    if summary:
        height, width = overlay.shape[:2]
        panel_height = 100
        cv2.rectangle(
            overlay,
            (0, height - panel_height),
            (width, height),
            (0, 0, 0),
            thickness=-1,
        )
        text_lines = [
            f"Engagement score: {summary.raw_score:.2f}",
            f"Smoothed score: {summary.smoothed_score:.2f}",
            f"Rolling average: {summary.rolling_average:.2f}",
        ]
        if summary.dominant_emotion:
            text_lines.append(f"Dominant emotion: {summary.dominant_emotion}")

        for index, text in enumerate(text_lines):
            cv2.putText(
                overlay,
                text,
                (10, height - panel_height + 25 + index * 20),
                FONT,
                0.6,
                (0, 255, 0),
                2,
            )

        if summary.emotion_distribution:
            emotions = list(summary.emotion_distribution.items())
            bar_width = int(width / max(len(emotions), 1))
            base_y = height - 5
            for idx, (emotion, value) in enumerate(emotions):
                bar_height = int(60 * value)
                top_left = (idx * bar_width + 10, base_y - bar_height)
                bottom_right = (idx * bar_width + bar_width - 10, base_y)
                cv2.rectangle(overlay, top_left, bottom_right, (0, 128, 255), -1)
                cv2.putText(
                    overlay,
                    f"{emotion} {_format_percentage(value)}",
                    (top_left[0], top_left[1] - 5),
                    FONT,
                    0.5,
                    (0, 200, 255),
                    1,
                )

    return overlay
