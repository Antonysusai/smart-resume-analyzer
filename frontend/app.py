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

API_URL = "http://localhost:8000"

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
                result = response.json()
                _render_results(result)
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
                result = response.json()
                _render_batch_results(result)
            else:
                st.error(f"Error: {response.text}")


# ──────────────────────────────────────────────────────────────────────────────
# Result rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def _render_results(result: dict):
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

    st.divider()

    # ── Candidate Info ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Scores", "👤 Profile", "🛠 Skills", "💡 Insights"])

    with tab1:
        _render_score_charts(score)

    with tab2:
        _render_profile(entities, file_info)

    with tab3:
        _render_skills(entities, score, skills_by_cat)

    with tab4:
        _render_insights(score)


def _render_score_charts(score: dict):
    col1, col2 = st.columns(2)

    with col1:
        # Radar chart
        categories = ["Skills", "Experience", "Education", "Completeness", "Semantic Fit"]
        values = [
            score["skill_match_score"],
            score["experience_score"],
            score["education_score"],
            score["completeness_score"],
            score["semantic_similarity"] * 100,
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
            marker_color=["#0d6efd", "#198754", "#ffc107", "#0dcaf0", "#6f42c1"],
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
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💪 Strengths")
        for s in score.get("strengths", []):
            st.markdown(f'<p class="strength-item">✔ {s}</p>', unsafe_allow_html=True)
        if not score.get("strengths"):
            st.info("No major strengths detected.")

    with col2:
        st.subheader("⚠ Gaps")
        for g in score.get("gaps", []):
            st.markdown(f'<p class="gap-item">✘ {g}</p>', unsafe_allow_html=True)
        if not score.get("gaps"):
            st.success("No significant gaps found!")

    st.subheader("📋 Recommendations")
    for i, rec in enumerate(score.get("recommendations", []), 1):
        st.markdown(f"{i}. {rec}")


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
