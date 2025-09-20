"""Engagement tracking utilities."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from statistics import mean
from typing import Deque, Dict, Iterable, List, Optional

from .analyzer import EmotionDetection
from .config import TrackerConfig


@dataclass
class FrameSummary:
    """Summary statistics computed for a single frame."""

    raw_score: float
    smoothed_score: float
    rolling_average: float
    dominant_emotion: Optional[str]
    emotion_distribution: Dict[str, float]


class EngagementTracker:
    """Aggregate engagement scores across frames."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._scores: Deque[float] = deque(maxlen=self.config.history_size)
        self._smoothed_score: Optional[float] = None

    def _score_for_emotions(self, emotions: Dict[str, float]) -> float:
        total_intensity = sum(emotions.values())
        if total_intensity <= 0:
            return 0.0

        weighted_total = 0.0
        for emotion, intensity in emotions.items():
            weight = self.config.weight_for(emotion.lower())
            weighted_total += weight * intensity
        return weighted_total / total_intensity

    def update(self, detections: Iterable[EmotionDetection]) -> FrameSummary:
        """Update the tracker with the latest detections."""

        frame_scores: List[float] = []
        emotion_totals: Counter[str] = Counter()

        for detection in detections:
            emotions = detection.emotions
            if not emotions:
                continue
            score = self._score_for_emotions(emotions)
            frame_scores.append(score)
            for emotion, value in emotions.items():
                emotion_totals[emotion.lower()] += value

        if frame_scores:
            raw_score = sum(frame_scores) / len(frame_scores)
        else:
            raw_score = 0.0

        self._scores.append(raw_score)

        if self._smoothed_score is None:
            self._smoothed_score = raw_score
        else:
            alpha = self.config.smoothing_factor
            self._smoothed_score = (1 - alpha) * self._smoothed_score + alpha * raw_score

        rolling_average = mean(self._scores) if self._scores else 0.0

        total_emotion_intensity = sum(emotion_totals.values())
        if total_emotion_intensity > 0:
            distribution = {
                emotion: value / total_emotion_intensity
                for emotion, value in emotion_totals.items()
            }
            dominant_emotion = max(distribution, key=distribution.get)
        else:
            distribution = {}
            dominant_emotion = None

        return FrameSummary(
            raw_score=raw_score,
            smoothed_score=self._smoothed_score or 0.0,
            rolling_average=rolling_average,
            dominant_emotion=dominant_emotion,
            emotion_distribution=distribution,
        )
