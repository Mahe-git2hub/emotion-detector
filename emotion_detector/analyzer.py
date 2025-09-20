"""Emotion detection utilities built on top of the :mod:`fer` library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from fer import FER
except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing at runtime
    raise ImportError(
        "The 'fer' package is required for emotion detection. Install dependencies "
        "with `pip install -r requirements.txt`."
    ) from exc


@dataclass
class EmotionDetection:
    """Represents a single detected face and the associated emotions."""

    box: Tuple[int, int, int, int]
    emotions: Dict[str, float]

    @property
    def dominant_emotion(self) -> Optional[str]:
        if not self.emotions:
            return None
        return max(self.emotions, key=self.emotions.get)


class EmotionAnalyzer:
    """Thin wrapper around :class:`fer.FER` to ease integration."""

    def __init__(self, use_mtcnn: bool = True) -> None:
        self._detector = FER(mtcnn=use_mtcnn)

    def detect(self, frame) -> List[EmotionDetection]:  # type: ignore[override]
        """Detect emotions in a frame.

        Parameters
        ----------
        frame:
            A numpy array representing the RGB frame.
        """

        results = self._detector.detect_emotions(frame)
        detections: List[EmotionDetection] = []
        for result in results:
            box = tuple(int(value) for value in result.get("box", (0, 0, 0, 0)))
            emotions = {
                emotion.lower(): float(score) for emotion, score in result.get("emotions", {}).items()
            }
            detections.append(EmotionDetection(box=box, emotions=emotions))
        return detections


def format_detection_summary(detections: Iterable[EmotionDetection]) -> Dict[str, float]:
    """Return average confidence for each detected emotion in the frame."""

    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for detection in detections:
        for emotion, score in detection.emotions.items():
            totals[emotion] = totals.get(emotion, 0.0) + score
            counts[emotion] = counts.get(emotion, 0) + 1

    if not totals:
        return {}

    return {emotion: totals[emotion] / counts[emotion] for emotion in totals}
