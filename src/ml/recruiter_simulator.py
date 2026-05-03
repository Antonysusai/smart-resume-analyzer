"""
Six-second recruiter scan simulation for German hiring workflows.
"""

from src.ml.achievement_detector import AchievementAnalysis
from src.ml.hard_filter import HardFilterEngine, HardFilterResult, JobRequirements
from src.nlp.extractor import ResumeEntities


class RecruiterSimulator:
    """Produces quick human-readable signals a recruiter would notice first."""

    def scan(
        self,
        entities: ResumeEntities,
        requirements: JobRequirements,
        matched_skills: list[str],
        missing_skills: list[str],
        achievement_analysis: AchievementAnalysis,
        hard_filter_result: HardFilterResult,
    ) -> dict[str, object]:
        first_impression = []
        red_flags = []
        strengths = []

        if entities.name:
            first_impression.append(f"Candidate profile parsed for {entities.name}")
        if entities.total_experience_years:
            first_impression.append(f"{entities.total_experience_years:g} years of experience detected")
        if matched_skills:
            first_impression.append("Relevant stack appears early: " + ", ".join(matched_skills[:4]))

        if hard_filter_result.rejection_reason:
            red_flags.extend(hard_filter_result.failed_criteria)
        if requirements.german_required and not HardFilterEngine.best_german_level(entities.languages):
            red_flags.append("No German language level mentioned")
        if achievement_analysis.quantified_count == 0:
            red_flags.append("No measurable achievements found")
        if len(missing_skills) >= 3:
            red_flags.append("Several job skills are missing from the CV")
        if not entities.linkedin:
            red_flags.append("LinkedIn profile missing")

        if len(matched_skills) >= 3:
            strengths.append("Strong skill alignment: " + ", ".join(matched_skills[:5]))
        if achievement_analysis.quantified_count:
            strengths.append(f"{achievement_analysis.quantified_count} quantified achievement signal(s)")
        german_level = HardFilterEngine.best_german_level(entities.languages)
        if german_level:
            strengths.append(f"German language level found: {german_level}")
        if entities.education:
            strengths.append("Education section is present")

        shortlist_score = 0.45
        shortlist_score += min(len(matched_skills), 6) * 0.05
        shortlist_score += min(entities.total_experience_years, 8) * 0.025
        shortlist_score += 0.12 if achievement_analysis.has_metrics else -0.10
        shortlist_score += 0.08 if HardFilterEngine.best_german_level(entities.languages) else 0.0
        shortlist_score -= min(len(red_flags), 5) * 0.08
        shortlist_probability = round(max(0.0, min(shortlist_score, 1.0)), 4)
        if shortlist_probability >= 0.75:
            decision = "shortlist"
        elif shortlist_probability >= 0.55:
            decision = "consider"
        else:
            decision = "reject"

        return {
            "first_impression": first_impression[:5],
            "red_flags": red_flags[:6],
            "strengths": strengths[:6],
            "shortlist_probability": shortlist_probability,
            "recruiter_decision": {
                "decision": decision,
                "confidence": shortlist_probability,
                "time_to_decision_sec": 6,
            },
        }
