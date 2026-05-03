"""
Achievement detection for recruiter-style CV evaluation.
"""

from dataclasses import dataclass, field
import re


ACTION_VERBS = {
    "built", "developed", "created", "implemented", "launched", "designed",
    "improved", "optimized", "reduced", "increased", "automated", "delivered",
    "led", "owned", "scaled", "migrated", "deployed", "streamlined",
    "entwickelt", "verbessert", "optimiert", "reduziert", "gesteigert",
}


@dataclass
class AchievementSignal:
    text: str
    has_metric: bool
    action_verbs: list[str] = field(default_factory=list)


@dataclass
class AchievementAnalysis:
    score: float
    quantified_count: int
    action_verb_count: int
    has_metrics: bool = False
    examples: list[str] = field(default_factory=list)


class AchievementDetector:
    """Finds quantified impact and action-oriented accomplishment statements."""

    METRIC_PATTERN = re.compile(
        r"(\b\d+(?:[.,]\d+)?\s*(?:%|percent|k|m|x|eur|€|usd|\$|hours?|days?|weeks?|months?|users?|customers?|records?|million|billion|revenue|costs?)\b|\b\d+(?:[.,]\d+)?\s*%)",
        re.IGNORECASE,
    )

    def analyze(self, text: str) -> AchievementAnalysis:
        lines = self._candidate_lines(text)
        signals = []
        for line in lines:
            lower = line.lower()
            verbs = sorted({verb for verb in ACTION_VERBS if re.search(rf"\b{re.escape(verb)}\b", lower)})
            has_metric = bool(self.METRIC_PATTERN.search(line))
            if verbs or has_metric:
                signals.append(AchievementSignal(text=line, has_metric=has_metric, action_verbs=verbs))

        quantified = sum(1 for signal in signals if signal.has_metric)
        action_count = sum(len(signal.action_verbs) for signal in signals)
        score = min(100.0, quantified * 28 + min(action_count, 8) * 7)
        if quantified and action_count:
            score = min(100.0, score + 10)

        return AchievementAnalysis(
            score=round(score, 1),
            quantified_count=quantified,
            action_verb_count=action_count,
            has_metrics=quantified > 0,
            examples=[signal.text for signal in signals[:5]],
        )

    def _candidate_lines(self, text: str) -> list[str]:
        lines = []
        for raw in re.split(r"[\n\r]+|(?<=[.!?])\s+", text or ""):
            line = raw.strip(" -•\t")
            if 18 <= len(line) <= 220:
                lines.append(line)
        return lines[:120]
