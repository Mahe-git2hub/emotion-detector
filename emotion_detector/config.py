"""Configuration utilities for the real-time emotion detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping


@dataclass(frozen=True)
class EmotionWeights:
    """Weight configuration for mapping emotions to engagement scores.

    Positive emotions contribute a value of ``+1`` to the engagement score,
    neutral emotions contribute ``+0.5``, and negative emotions contribute ``0``.
    The mapping can be customised at runtime by instantiating this dataclass with
    different weight collections.
    """

    positive: Iterable[str] = field(
        default_factory=lambda: ("happy", "surprise", "excited", "engaged")
    )
    neutral: Iterable[str] = field(default_factory=lambda: ("neutral", "calm"))
    negative: Iterable[str] = field(
        default_factory=lambda: (
            "sad",
            "angry",
            "disgust",
            "fear",
            "bored",
            "contempt",
            "tired",
        )
    )

    def as_lookup(self) -> Dict[str, float]:
        """Return a dictionary mapping emotions to their engagement weight."""

        lookup: Dict[str, float] = {}
        for emotion in self.positive:
            lookup[emotion] = 1.0
        for emotion in self.neutral:
            lookup[emotion] = 0.5
        for emotion in self.negative:
            lookup[emotion] = 0.0
        return lookup


@dataclass(frozen=True)
class TrackerConfig:
    """Configuration for engagement tracking."""

    smoothing_factor: float = 0.2
    history_size: int = 150
    emotion_weights: EmotionWeights = field(default_factory=EmotionWeights)

    def weight_for(self, emotion: str) -> float:
        """Return the engagement contribution for a specific emotion."""

        weights = self.emotion_weights.as_lookup()
        # Fallback weight for emotions that are not explicitly mapped.
        return weights.get(emotion, 0.5)


def merge_emotion_weights(
    base: EmotionWeights, overrides: Mapping[str, Iterable[str]] | None
) -> EmotionWeights:
    """Merge user supplied emotion categories into the default configuration."""

    if not overrides:
        return base

    def normalise(values: Iterable[str]) -> Iterable[str]:
        return tuple(sorted({v.lower() for v in values}))

    positive = overrides.get("positive", base.positive)
    neutral = overrides.get("neutral", base.neutral)
    negative = overrides.get("negative", base.negative)

    return EmotionWeights(
        positive=normalise(positive),
        neutral=normalise(neutral),
        negative=normalise(negative),
    )
