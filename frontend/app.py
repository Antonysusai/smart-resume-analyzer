"""
Streamlit Frontend — Smart Resume Analyzer
Interactive UI for uploading resumes and visualizing analysis results.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .grade-badge {
        display: inline-block;
        padding: 0.4em 1em;
        border-radius: 8px;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .grade-A { background:#d4edda; color:#155724; }
    .grade-B { background:#cce5ff; color:#004085; }
    .grade-C { background:#fff3cd; color:#856404; }
    .grade-D { background:#f8d7da; color:#721c24; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #0d6efd;
    }
    .skill-chip {
        display: inline-block;
        background: #e7f3ff;
        color: #0d6efd;
        border-radius: 20px;
        padding: 3px 12px;
        margin: 3px;
        font-size: 0.85rem;
    }
    .missing-chip {
        display: inline-block;
        background: #fdecea;
        color: #c0392b;
        border-radius: 20px;
        padding: 3px 12px;
        margin: 3px;
        font-size: 0.85rem;
    }
    .strength-item { color: #27ae60; }
    .gap-item { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/resume.png", width=64)
    st.title("Smart Resume Analyzer")
    st.caption("CV + NLP + ML powered analysis")
    st.divider()

    mode = st.radio("Mode", ["Single Resume", "Batch Ranking"])
    st.divider()

    st.markdown("**API Status**")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success("API Online ✓")
        else:
            st.error("API Error")
    except Exception:
        st.error("API Offline — start the server:\n`uvicorn src.api.main:app`")

# ──────────────────────────────────────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────────────────────────────────────

st.title("📄 Smart Resume Analyzer")
st.markdown("Upload a resume and paste a job description to get an AI-powered analysis.")
st.divider()

# ── Job Description Input ──────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    job_description = st.text_area(
        "Job Description",
        placeholder="Paste the job description here...",
        height=200,
    )

with col2:
    job_skills_input = st.text_area(
        "Required Skills (one per line or comma-separated)",
        placeholder="Python\nMachine Learning\nSQL\nAWS",
        height=200,
    )

# ── File Upload ────────────────────────────────────────────────────────────────
st.divider()

if mode == "Single Resume":
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF, PNG, JPG, JPEG)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    if uploaded_file and job_description:
        if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
            with st.spinner("Processing your resume — running CV, NLP & ML pipeline..."):
                job_skills_str = ",".join(
                    s.strip()
                    for s in job_skills_input.replace("\n", ",").split(",")
                    if s.strip()
                )
                response = requests.post(
                    f"{API_URL}/analyze",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    data={"job_description": job_description, "job_skills": job_skills_str},
                    timeout=120,
                )

            if response.status_code == 200:
                st.session_state["single_result"] = response.json()
            else:
                st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")
    elif not job_description:
        st.info("Please paste a job description to enable analysis.")

else:
    uploaded_files = st.file_uploader(
        "Upload Multiple Resumes (max 20)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded_files and job_description:
        if st.button("🏆 Rank Candidates", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {len(uploaded_files)} resumes..."):
                job_skills_str = ",".join(
                    s.strip()
                    for s in job_skills_input.replace("\n", ",").split(",")
                    if s.strip()
                )
                files = [
                    ("files", (f.name, f.getvalue(), f.type))
                    for f in uploaded_files
                ]
                response = requests.post(
                    f"{API_URL}/batch-analyze",
                    files=files,
                    data={"job_description": job_description, "job_skills": job_skills_str},
                    timeout=300,
                )

            if response.status_code == 200:
                st.session_state["batch_result"] = response.json()
            else:
                st.error(f"Error: {response.text}")


# ──────────────────────────────────────────────────────────────────────────────
# Result rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def _render_results(result: dict):
    if result.get("status") == "rejected" and "score" not in result:
        st.divider()
        st.error(f"Rejected by hard filter: {result.get('rejection_reason', 'Requirement not met')}")
        if result.get("failed_criteria"):
            st.subheader("Failed Criteria")
            for item in result["failed_criteria"]:
                st.markdown(f"- {item}")
        if result.get("critical_missing_skills"):
            st.subheader("Critical Missing Skills")
            st.write(", ".join(result["critical_missing_skills"]))
        if result.get("risk_flags"):
            st.subheader("Risk Flags")
            for item in result["risk_flags"]:
                st.markdown(f"- {item}")
        if result.get("diagnostics"):
            st.subheader("Diagnostics")
            st.json(result["diagnostics"])
        return

    entities = result["entities"]
    score = result["score"]
    file_info = result["file_info"]
    skills_by_cat = result["skills_by_category"]

    # ── Header ────────────────────────────────────────────────────────────────
    st.divider()
    h1, h2, h3, h4 = st.columns(4)

    grade = score["grade"]
    with h1:
        st.markdown(
            f'<div class="grade-badge grade-{grade}">{grade}</div>'
            f'<br><small>Overall Grade</small>',
            unsafe_allow_html=True,
        )
    with h2:
        st.metric("Overall Score", f"{score['overall_score']}/100")
    with h3:
        st.metric("Fit Probability", f"{score['fit_probability']*100:.1f}%")
    with h4:
        st.metric("Processing Time", f"{result['processing_time_ms']:.0f}ms")

    confidence = result.get("confidence") or score.get("confidence")
    if confidence:
        st.caption(f"Confidence: {confidence['level'].title()} ({confidence['score']*100:.1f}%)")

    if score.get("status") == "rejected":
        st.error(f"Rejected by hard filter: {score.get('rejection_reason', 'Requirement not met')}")
    else:
        st.success("Passed hard filters and moved to scoring")

    st.divider()

    # ── German ATS + Recruiter flow ───────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "ATS Gate", "Score", "Recruiter View", "Improvement Plan", "Summary"
    ])

    with tab1:
        st.subheader("ATS Gate")
        st.success("Accepted: passed hard filters")
        st.write(f"**Reason:** {result.get('hiring_decision', {}).get('reason', 'Meets core hard-filter requirements')}")
        _render_profile(entities, file_info)

    with tab2:
        _render_score_charts(score)

    with tab3:
        _render_insights(score)

    with tab4:
        _render_improvement_plan(result, score, skills_by_cat)

    with tab5:
        _render_summary(result, score)


def _render_score_charts(score: dict):
    col1, col2 = st.columns(2)
    breakdown = score.get("score_breakdown") or {
        "skills": score["skill_match_score"],
        "experience": score["experience_score"],
        "language": score.get("language_score", 0),
        "achievements": score.get("achievement_score", 0),
        "formatting": score.get("formatting_score", score.get("completeness_score", 0)),
        "recruiter_alignment": score.get("recruiter_alignment_score", 0),
    }

    with col1:
        # Radar chart
        categories = ["Skills", "Experience", "German", "Achievements", "Formatting", "Recruiter"]
        values = [
            breakdown["skills"],
            breakdown["experience"],
            breakdown["language"],
            breakdown["achievements"],
            breakdown["formatting"],
            breakdown["recruiter_alignment"],
        ]
        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line_color="#0d6efd",
            fillcolor="rgba(13, 110, 253, 0.15)",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title="Score Breakdown",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Bar chart
        fig2 = go.Figure(go.Bar(
            x=values,
            y=categories,
            orientation="h",
            marker_color=["#0d6efd", "#198754", "#6f42c1", "#0dcaf0", "#ffc107", "#6c757d"],
            text=[f"{v:.1f}" for v in values],
            textposition="inside",
        ))
        fig2.update_layout(
            xaxis=dict(range=[0, 100]),
            title="Dimension Scores",
            height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # SHAP explanation (if available)
    if score.get("shap_explanation"):
        st.subheader("Feature Impact (SHAP)")
        shap_df = pd.DataFrame(
            list(score["shap_explanation"].items()),
            columns=["Feature", "SHAP Value"]
        ).sort_values("SHAP Value", key=abs, ascending=False).head(10)
        fig3 = px.bar(
            shap_df, x="SHAP Value", y="Feature", orientation="h",
            color="SHAP Value",
            color_continuous_scale=["#e74c3c", "#f8f9fa", "#27ae60"],
            title="Top 10 Features Driving the Fit Score",
        )
        st.plotly_chart(fig3, use_container_width=True)


def _render_profile(entities: dict, file_info: dict):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Contact Info")
        st.write(f"**Name:** {entities.get('name') or '—'}")
        st.write(f"**Email:** {entities.get('email') or '—'}")
        st.write(f"**Phone:** {entities.get('phone') or '—'}")
        st.write(f"**Location:** {entities.get('location') or '—'}")
        st.write(f"**LinkedIn:** {entities.get('linkedin') or '—'}")
        st.write(f"**GitHub:** {entities.get('github') or '—'}")
        st.write(f"**Languages:** {', '.join(entities.get('languages', [])) or '—'}")

        st.subheader("Document Info")
        st.write(f"**File:** {file_info['filename']}")
        st.write(f"**Pages:** {file_info['pages']}")
        st.write(f"**Scanned:** {'Yes' if file_info['is_scanned'] else 'No'}")
        st.write(f"**Quality Score:** {file_info['quality_score']:.2f}")

    with c2:
        st.subheader("Experience")
        for exp in entities.get("experience", []):
            st.markdown(
                f"**{exp['title']}** @ {exp['company']}  \n"
                f"{exp.get('start_date','?')} → {exp.get('end_date','?')}"
            )
            st.divider()

        st.subheader("Education")
        for edu in entities.get("education", []):
            st.markdown(f"**{edu['degree']}**  \n{edu['institution']} {edu.get('year','')}")

    if entities.get("summary"):
        st.subheader("Professional Summary")
        st.info(entities["summary"])


def _render_skills(entities: dict, score: dict, skills_by_cat: dict):
    st.subheader(f"✅ Matched Skills ({len(score['matched_skills'])})")
    chips = " ".join(
        f'<span class="skill-chip">{s}</span>' for s in score["matched_skills"]
    )
    st.markdown(chips or "_None matched_", unsafe_allow_html=True)

    st.subheader(f"❌ Missing Skills ({len(score['missing_skills'])})")
    missing = " ".join(
        f'<span class="missing-chip">{s}</span>' for s in score["missing_skills"]
    )
    st.markdown(missing or "_None missing_", unsafe_allow_html=True)

    if score.get("optional_missing_skills"):
        st.subheader(f"Optional Missing Skills ({len(score['optional_missing_skills'])})")
        optional = " ".join(
            f'<span class="missing-chip">{s}</span>' for s in score["optional_missing_skills"]
        )
        st.markdown(optional, unsafe_allow_html=True)

    if score.get("skill_gap_analysis"):
        st.subheader("Skill Gap Priority")
        st.dataframe(pd.DataFrame(score["skill_gap_analysis"]), use_container_width=True, hide_index=True)

    st.subheader("Skills by Category")
    if skills_by_cat:
        for cat, skills in skills_by_cat.items():
            with st.expander(f"{cat.replace('_', ' ').title()} ({len(skills)})"):
                st.write(", ".join(skills))
    else:
        st.info("No skills categorized.")

    if entities.get("certifications"):
        st.subheader("Certifications")
        for cert in entities["certifications"]:
            st.markdown(f"- {cert}")


def _render_insights(score: dict):
    insights = score.get("recruiter_insights", {})
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("First Impression")
        for item in insights.get("first_impression", []):
            st.markdown(f"- {item}")
        if not insights.get("first_impression"):
            st.info("No first impression signals detected.")

    with col2:
        st.subheader("Red Flags")
        for item in insights.get("red_flags", score.get("gaps", [])):
            st.markdown(f'<p class="gap-item">- {item}</p>', unsafe_allow_html=True)
        if not insights.get("red_flags") and not score.get("gaps"):
            st.success("No immediate red flags found.")

    with col3:
        st.subheader("Strength Signals")
        for item in insights.get("strengths", score.get("strengths", [])):
            st.markdown(f'<p class="strength-item">✔ {item}</p>', unsafe_allow_html=True)
        if not insights.get("strengths") and not score.get("strengths"):
            st.info("No major strengths detected.")

    if score.get("achievement_analysis", {}).get("examples"):
        st.subheader("Achievement Signals")
        for example in score["achievement_analysis"]["examples"]:
            st.markdown(f"- {example}")

    if "shortlist_probability" in insights:
        st.metric("Shortlist Probability", f"{insights['shortlist_probability']*100:.1f}%")

    if score.get("recruiter_decision"):
        st.subheader("Final Recruiter Decision")
        st.json(score["recruiter_decision"])

    if score.get("confidence"):
        st.subheader("Confidence")
        st.json(score["confidence"])

    if score.get("hiring_decision"):
        st.subheader("Hiring Decision")
        st.json(score["hiring_decision"])


def _render_improvement_plan(result: dict, score: dict, skills_by_cat: dict):
    st.subheader("Prioritized Recommendations")
    recommendations = result.get("recommendations") or score.get("recommendations", [])
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    if not recommendations:
        st.info("No recommendations generated.")

    if result.get("issues") or score.get("issues"):
        st.subheader("Severity Ranked Issues")
        st.dataframe(pd.DataFrame(result.get("issues") or score.get("issues")), use_container_width=True, hide_index=True)

    st.subheader("Skill Gaps")
    if score.get("skill_gap_analysis"):
        st.dataframe(pd.DataFrame(score["skill_gap_analysis"]), use_container_width=True, hide_index=True)
    else:
        st.success("No prioritized skill gaps detected.")

    if skills_by_cat:
        st.subheader("Extracted Skills by Category")
        for cat, skills in skills_by_cat.items():
            with st.expander(f"{cat.replace('_', ' ').title()} ({len(skills)})"):
                st.write(", ".join(skills))


def _render_summary(result: dict, score: dict):
    st.subheader("Job-Fit Summary")
    st.info(result.get("summary") or score.get("summary", "No summary generated."))
    if result.get("hiring_decision"):
        st.subheader("Final Hiring Verdict")
        st.json(result["hiring_decision"])
    if result.get("achievement_examples"):
        st.subheader("Achievement Examples")
        for example in result["achievement_examples"]:
            st.markdown(f"- {example}")


def _render_batch_results(result: dict):
    st.divider()
    st.subheader(f"🏆 Ranked Candidates ({result['total']} analyzed)")
    df = pd.DataFrame(result["ranked_candidates"])
    if "error" in df.columns:
        df = df.fillna({"error": ""})

    st.dataframe(
        df.style.background_gradient(subset=["overall_score"], cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True,
    )

    if "overall_score" in df.columns:
        fig = px.bar(
            df.dropna(subset=["overall_score"]),
            x="filename",
            y="overall_score",
            color="grade",
            title="Candidate Ranking by Overall Score",
            labels={"overall_score": "Score", "filename": "Candidate"},
        )
        st.plotly_chart(fig, use_container_width=True)


if mode == "Single Resume" and st.session_state.get("single_result"):
    _render_results(st.session_state["single_result"])
elif mode == "Batch Ranking" and st.session_state.get("batch_result"):
    _render_batch_results(st.session_state["batch_result"])
