from __future__ import annotations

import re
import time
import hashlib
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="UniAssist India", page_icon="🎓", layout="wide")

DATA_DIR = Path("data")
FEEDBACK_FILE = DATA_DIR / "feedback.csv"
QUERY_LOG_FILE = DATA_DIR / "query_logs.csv"
ALERTS_FILE = DATA_DIR / "alerts.csv"
TICKETS_FILE = DATA_DIR / "tickets.csv"

SIMILARITY_DEFAULT = 0.65
UNIVERSITIES = ["University 1", "University 2", "University 3"]

USERS = {
    "student_demo": {"password": "student123", "role": "student"},
    "teacher_demo": {"password": "teacher123", "role": "teacher"},
    "parent_demo": {"password": "parent123", "role": "parent"},
    "admin_demo": {"password": "admin123", "role": "developer_admin"},
}

SCOPE_KEYWORDS = {
    "attendance",
    "exam",
    "internship",
    "policy",
    "grades",
    "cgpa",
    "credit",
    "placement",
    "scholarship",
    "leave",
    "semester",
    "academic",
}

SAFE_FALLBACK = (
    "I do not have reliable evidence for this query in the current academic dataset. "
    "Please refer to official university notices or ask admin to add this policy source."
)

CATEGORY_STEPS = {
    "attendance": ["Check attendance ledger", "Submit regularization request", "Meet class advisor"],
    "internship": ["Verify eligibility", "Prepare documents", "Apply before deadline"],
    "exam": ["Confirm exam schedule", "Check appeal/revaluation rules", "Contact exam cell"],
    "general": ["Read official circular", "Follow department process", "Escalate to admin office"],
}

st.markdown(
    """
<style>
:root {
  --brand-1: #ff8f1f;
  --brand-2: #0f8a5f;
  --brand-3: #1f4ed8;
  --ink-1: #0f172a;
  --ink-2: #334155;
  --muted: #64748b;
  --line: #cbd5e1;
  --card: #ffffff;
}
.main-banner {
  border-radius: 18px;
  padding: 20px 22px;
  background:
    radial-gradient(circle at 10% 20%, rgba(255,143,31,0.18) 0, rgba(255,143,31,0.03) 35%),
    radial-gradient(circle at 95% 5%, rgba(31,78,216,0.16) 0, rgba(31,78,216,0.03) 30%),
    linear-gradient(120deg, rgba(255,255,255,0.96), rgba(245,250,255,0.96));
  border: 1px solid rgba(31,78,216,0.22);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
  margin-bottom: 16px;
  animation: riseIn 0.45s ease-out;
}
.main-title { font-size: 36px; font-weight: 800; color: var(--brand-3); letter-spacing: 0.2px; }
.main-sub { color: var(--ink-2); font-size: 15px; }
.answer-card {
  background: var(--card);
  border-left: 6px solid var(--brand-3);
  color: var(--ink-1) !important;
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.answer-card:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.11);
}
.metric-card {
  background: #fff;
  border: 1px solid #dbe4ef;
  border-radius: 12px;
  padding: 12px 14px;
  color: var(--ink-1) !important;
  box-shadow: 0 4px 12px rgba(2, 6, 23, 0.04);
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 100% 0%, rgba(31,78,216,0.06) 0, rgba(31,78,216,0) 35%),
    radial-gradient(circle at 0% 100%, rgba(15,138,95,0.06) 0, rgba(15,138,95,0) 35%),
    #f8fafc;
}

/* Fix text and input contrast issues */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
.stDateInput input,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] > div {
  background: #ffffff !important;
  color: var(--ink-1) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
  color: var(--muted) !important;
  opacity: 1 !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus,
.stDateInput input:focus {
  border: 1px solid var(--brand-3) !important;
  box-shadow: 0 0 0 3px rgba(31, 78, 216, 0.12) !important;
}

label[data-testid="stWidgetLabel"] p {
  color: var(--ink-1) !important;
  font-weight: 600 !important;
}

.stButton > button,
[data-testid="baseButton-secondary"] {
  border-radius: 10px !important;
  border: 1px solid rgba(31, 78, 216, 0.35) !important;
  background: linear-gradient(90deg, #1f4ed8, #1550b8) !important;
  color: #ffffff !important;
  transition: transform 0.15s ease, box-shadow 0.2s ease !important;
  box-shadow: 0 6px 16px rgba(31, 78, 216, 0.24);
}

.stButton > button:hover,
[data-testid="baseButton-secondary"]:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 18px rgba(31, 78, 216, 0.26) !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
}

.stTabs [data-baseweb="tab"] {
  border-radius: 10px;
  padding: 8px 14px;
  border: 1px solid #dbe4ef;
  background: #ffffff;
  color: var(--ink-2);
}

.stTabs [aria-selected="true"] {
  color: #ffffff !important;
  background: linear-gradient(90deg, #1f4ed8, #1550b8) !important;
  border-color: #1f4ed8 !important;
}

[data-testid="stForm"] {
  border: 1px solid #e2e8f0 !important;
  border-radius: 14px !important;
  background: #ffffff !important;
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
}

@keyframes riseIn {
  from { transform: translateY(8px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}
</style>
""",
    unsafe_allow_html=True,
)


def ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not FEEDBACK_FILE.exists():
        pd.DataFrame(
            columns=[
                "timestamp",
                "user",
                "role",
                "university",
                "query",
                "response",
                "confidence",
                "feedback",
                "rating",
                "comment",
            ]
        ).to_csv(FEEDBACK_FILE, index=False)
    else:
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        required_feedback_cols = [
            "timestamp",
            "user",
            "role",
            "university",
            "query",
            "response",
            "confidence",
            "feedback",
            "rating",
            "comment",
        ]
        changed = False
        for col in required_feedback_cols:
            if col not in feedback_df.columns:
                feedback_df[col] = ""
                changed = True
        if changed:
            feedback_df = feedback_df[required_feedback_cols]
            feedback_df.to_csv(FEEDBACK_FILE, index=False)
    if not QUERY_LOG_FILE.exists():
        pd.DataFrame(
            columns=[
                "timestamp",
                "user",
                "role",
                "university",
                "department",
                "semester",
                "query",
                "category",
                "confidence",
                "latency_ms",
                "escalated",
            ]
        ).to_csv(QUERY_LOG_FILE, index=False)
    if not ALERTS_FILE.exists():
        pd.DataFrame(columns=["timestamp", "user", "alert_type", "details"]).to_csv(ALERTS_FILE, index=False)
    if not TICKETS_FILE.exists():
        pd.DataFrame(columns=["timestamp", "user", "role", "query", "priority", "status"]).to_csv(TICKETS_FILE, index=False)


def append_row(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False)


def compute_trust_score(confidence: float, citations_count: int, freshness_days: int) -> float:
    conf = max(0.0, min(1.0, confidence)) * 0.65
    cite = min(citations_count / 3, 1.0) * 0.2
    fresh = max(0.0, 1 - min(freshness_days, 365) / 365) * 0.15
    return round((conf + cite + fresh) * 100, 1)


def projected_attendance(current_pct: float, classes_done: int, future_classes: int, attend_future: int) -> float:
    total_classes = classes_done + future_classes
    if total_classes <= 0:
        return current_pct
    attended_now = (current_pct / 100.0) * classes_done
    return round(((attended_now + attend_future) / total_classes) * 100, 2)


def projected_cgpa(current_cgpa: float, credits_done: int, future_credits: int, expected_gp: float) -> float:
    total_credits = credits_done + future_credits
    if total_credits <= 0:
        return current_cgpa
    total_points = (current_cgpa * credits_done) + (expected_gp * future_credits)
    return round(total_points / total_credits, 2)


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def infer_category(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["attendance", "leave", "absent"]):
        return "attendance"
    if any(k in t for k in ["internship", "placement", "offer"]):
        return "internship"
    if any(k in t for k in ["exam", "grade", "revaluation", "cgpa"]):
        return "exam"
    return "general"


def map_university_labels(df: pd.DataFrame) -> pd.DataFrame:
    mapped = df.copy()
    if "university" not in mapped.columns:
        mapped["university"] = "University 1"
        return mapped

    unique_vals = [u for u in mapped["university"].dropna().astype(str).unique().tolist()]
    if not unique_vals:
        mapped["university"] = "University 1"
        return mapped

    mapping = {}
    for idx, name in enumerate(sorted(unique_vals)):
        mapping[name] = UNIVERSITIES[idx % len(UNIVERSITIES)]
    mapped["university"] = mapped["university"].astype(str).map(mapping).fillna("University 1")
    return mapped


@st.cache_data
def load_data() -> pd.DataFrame:
    frame = pd.read_csv("UniAssist_training_data.csv")
    if "question" not in frame.columns or "answer" not in frame.columns:
        raise ValueError("UniAssist_training_data.csv must contain 'question' and 'answer' columns")

    frame = map_university_labels(frame)
    frame["question"] = frame["question"].astype(str)
    frame["answer"] = frame["answer"].astype(str)
    if "category" not in frame.columns:
        frame["category"] = frame["question"].apply(infer_category)
    if "source" not in frame.columns:
        frame["source"] = "UniAssist_training_data.csv"
    if "last_updated" not in frame.columns:
        frame["last_updated"] = "2026-02-01"
    if "policy_link" not in frame.columns:
        frame["policy_link"] = "https://www.ugc.gov.in/"
    return frame


@st.cache_resource
def load_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_embeddings(questions: tuple[str, ...]) -> np.ndarray:
    return load_model().encode(list(questions), normalize_embeddings=True)


def active_kb() -> pd.DataFrame:
    base = load_data()
    extra = st.session_state.get("extra_sources", pd.DataFrame())
    if extra.empty:
        return base
    combined = pd.concat([base, extra], ignore_index=True)
    return combined.drop_duplicates(subset=["question", "answer"], keep="last")


def retrieve_dataset_answer(query: str, kb_df: pd.DataFrame, university: str, category_filter: str, top_k: int) -> dict:
    # Core logic preserved: dataset retrieval using embeddings + cosine similarity + threshold gating.
    filtered = kb_df[kb_df["university"].isin([university, "University 1"])].copy()
    if category_filter != "All":
        filtered = filtered[filtered["category"] == category_filter]
    if filtered.empty:
        filtered = kb_df.copy()

    questions = filtered["question"].tolist()
    embeddings = get_embeddings(tuple(questions))

    start = time.perf_counter()
    query_vec = load_model().encode([query], normalize_embeddings=True)
    semantic_scores = cosine_similarity(query_vec, embeddings)[0]
    latency_ms = int((time.perf_counter() - start) * 1000)

    q_tokens = tokenize(query)
    keyword_scores = np.array([len(q_tokens & tokenize(q)) / max(len(q_tokens), 1) for q in questions])
    bonus = np.where(filtered["category"].values == infer_category(query), 0.05, 0.0)
    final_scores = (semantic_scores * 0.72) + (keyword_scores * 0.23) + bonus

    ranked = filtered.copy()
    ranked["score"] = final_scores
    ranked = ranked.sort_values("score", ascending=False).head(top_k)

    if ranked.empty:
        return {
            "answer": SAFE_FALLBACK,
            "confidence": 0.0,
            "category": "general",
            "citations": [],
            "matched_question": None,
            "latency_ms": latency_ms,
            "freshness_days": 365,
        }

    best = ranked.iloc[0]
    best_updated = pd.to_datetime(best.get("last_updated", "2026-02-01"), errors="coerce")
    freshness_days = 365 if pd.isna(best_updated) else max(0, (datetime.now() - best_updated.to_pydatetime()).days)
    citations = []
    for _, row in ranked.iterrows():
        citations.append(
            {
                "source": str(row.get("source", "unknown")),
                "updated": str(row.get("last_updated", "unknown")),
                "link": str(row.get("policy_link", "https://www.ugc.gov.in/")),
            }
        )

    return {
        "answer": str(best["answer"]),
        "confidence": float(best["score"]),
        "category": str(best.get("category", "general")),
        "citations": citations,
        "matched_question": str(best["question"]),
        "latency_ms": latency_ms,
        "freshness_days": freshness_days,
    }


def init_session() -> None:
    defaults = {
        "active_user": "guest",
        "role_sessions": {},
        "extra_sources": pd.DataFrame(),
        "review_queue": [],
        "resolved_reviews": [],
        "last_response": None,
        "conversation": [],
        "checklist": [
            {"task": "Update resume", "due": str(date.today()), "done": False},
            {"task": "Check attendance", "due": str(date.today()), "done": False},
        ],
        "similarity_threshold": SIMILARITY_DEFAULT,
        "verified_mode": True,
        "consent": True,
        "session_start": time.time(),
        "policy_hashes": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def feedback_snapshot() -> tuple[float, int]:
    if not FEEDBACK_FILE.exists():
        return 0.0, 0
    df = pd.read_csv(FEEDBACK_FILE)
    if df.empty or "rating" not in df.columns:
        return 0.0, 0
    rating_series = pd.to_numeric(df["rating"], errors="coerce").dropna()
    if rating_series.empty:
        return 0.0, 0
    return float(rating_series.mean()), int(rating_series.count())


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("Configuration")
        st.selectbox("University", UNIVERSITIES, key="selected_university")
        st.slider("Similarity threshold", 0.50, 0.90, SIMILARITY_DEFAULT, 0.01, key="similarity_threshold")
        st.toggle("Verified answer mode", key="verified_mode")
        st.checkbox("Allow analytics logging", key="consent")

        avg_rating, rating_count = feedback_snapshot()
        st.caption(f"Region: India (IST)")
        st.caption(f"Session uptime: {int((time.time() - st.session_state['session_start']) / 60)} min")
        st.caption(f"Global rating: {avg_rating:.1f}/5 from {rating_count} responses")

        st.divider()
        st.subheader("Author")
        st.write("Mehul Kumar")
        st.write("UniAssist India Project")

        st.divider()
        st.caption("© 2026 UniAssist India. All rights reserved.")


def role_login_gate(role: str, label: str) -> tuple[bool, str]:
    session_roles = st.session_state["role_sessions"]
    if role in session_roles:
        user = session_roles[role]
        st.success(f"{label}: logged in as {user}")
        if st.button(f"Logout {label}", key=f"logout_{role}"):
            session_roles.pop(role, None)
            st.rerun()
        return True, user

    st.info(f"{label}: login required")
    with st.form(f"login_{role}"):
        username = st.text_input("Username", key=f"user_{role}")
        password = st.text_input("Password", type="password", key=f"pass_{role}")
        submit = st.form_submit_button("Login")
        if submit:
            rec = USERS.get(username)
            if rec and rec["password"] == password and rec["role"] == role:
                session_roles[role] = username
                st.session_state["active_user"] = username
                st.success("Login successful")
                st.rerun()
            st.error("Invalid role credentials")
    return False, "guest"


def in_scope(query: str) -> bool:
    return len(tokenize(query) & SCOPE_KEYWORDS) > 0


def log_query(query: str, result: dict, dept: str, sem: int, escalated: bool, user: str, role: str) -> None:
    if not st.session_state["consent"]:
        return
    append_row(
        QUERY_LOG_FILE,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user": user,
            "role": role,
            "university": st.session_state["selected_university"],
            "department": dept,
            "semester": sem,
            "query": query,
            "category": result["category"],
            "confidence": round(result["confidence"], 4),
            "latency_ms": result["latency_ms"],
            "escalated": int(escalated),
        },
    )


def assistant_tab(kb_df: pd.DataFrame, key_prefix: str, user: str, role: str) -> None:
    st.subheader("Academic Assistant")
    c1, c2, c3 = st.columns(3)
    with c1:
        dept = st.selectbox("Department", ["CSE", "ECE", "ME", "CE"], key=f"{key_prefix}_department")
    with c2:
        sem = st.selectbox("Semester", list(range(1, 9)), key=f"{key_prefix}_semester")
    with c3:
        category_filter = st.selectbox(
            "Category",
            ["All", "attendance", "exam", "internship", "general"],
            key=f"{key_prefix}_assistant_category",
        )

    with st.form(f"qa_form_{key_prefix}"):
        query = st.text_area(
            "Ask your query",
            placeholder="Example: minimum attendance for semester exams",
            key=f"{key_prefix}_query",
        )
        top_k = st.slider("Top sources", 1, 5, 3, key=f"{key_prefix}_assistant_top_k")
        asked = st.form_submit_button("Get answer")

    if asked:
        query = query.strip()
        if not query:
            st.warning("Please enter a query.")
            return
        if not in_scope(query):
            st.error("Out of scope. UniAssist currently handles academic and internship queries.")
            return

        result = retrieve_dataset_answer(query, kb_df, st.session_state["selected_university"], category_filter, top_k)
        escalated = result["confidence"] < st.session_state["similarity_threshold"]
        trust_score = compute_trust_score(
            result["confidence"], len(result["citations"]), int(result.get("freshness_days", 365))
        )

        shown_answer = result["answer"] if not escalated else SAFE_FALLBACK
        if escalated:
            st.session_state["review_queue"].append(
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "query": query,
                    "suggested_answer": result["answer"],
                    "confidence": result["confidence"],
                }
            )

        st.markdown("<div class='answer-card'>" + shown_answer + "</div>", unsafe_allow_html=True)
        st.caption(
            f"Confidence: {result['confidence']:.2f} | Trust score: {trust_score}/100 | Latency: {result['latency_ms']} ms"
        )
        st.caption(f"Matched query: {result['matched_question']}")

        st.markdown("### Action Steps")
        for step in CATEGORY_STEPS.get(result["category"], CATEGORY_STEPS["general"]):
            st.write(f"- {step}")

        if st.session_state["verified_mode"]:
            st.markdown("### Citations")
            for cite in result["citations"]:
                st.write(f"- {cite['source']} | {cite['updated']} | {cite['link']}")

        st.session_state["last_response"] = {
            "query": query,
            "response": shown_answer,
            "confidence": result["confidence"],
            "user": user,
            "role": role,
        }
        st.session_state["conversation"].append({"q": query, "a": shown_answer})
        log_query(query, result, dept, sem, escalated, user=user, role=role)

    if st.button("Escalate to academic office", key=f"{key_prefix}_escalate_btn"):
        lr = st.session_state.get("last_response")
        if not lr:
            st.warning("Ask at least one question before escalating.")
        else:
            append_row(
                TICKETS_FILE,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "user": lr.get("user", user),
                    "role": lr.get("role", role),
                    "query": lr["query"],
                    "priority": "high" if lr["confidence"] < st.session_state["similarity_threshold"] else "normal",
                    "status": "open",
                },
            )
            st.success("Escalation ticket created")

    with st.expander("Conversation memory"):
        history = st.session_state["conversation"][-8:]
        if not history:
            st.caption("No conversation yet.")
        else:
            for turn in history:
                st.write(f"Q: {turn['q']}")
                st.write(f"A: {turn['a']}")
                st.write("---")


def teacher_tab(user: str) -> None:
    st.subheader("Teacher Desk")
    st.caption(f"Logged in as {user}")
    st.caption("Publish verified internship/academic opportunities as additional sources.")
    with st.form("teacher_source_form"):
        program = st.text_input("Program / Opportunity title", key="teacher_program")
        eligibility = st.text_input("Eligibility summary", key="teacher_eligibility")
        deadline = st.date_input("Deadline", value=date.today(), key="teacher_deadline")
        submit = st.form_submit_button("Publish source")
        if submit and program.strip() and eligibility.strip():
            row = pd.DataFrame(
                [
                    {
                        "question": f"How to apply for {program}?",
                        "answer": f"Eligibility: {eligibility}. Deadline: {deadline}. Apply via placement cell.",
                        "university": st.session_state["selected_university"],
                        "category": "internship",
                        "source": "teacher_portal",
                        "last_updated": str(date.today()),
                        "policy_link": "https://www.aicte-india.org/",
                    }
                ]
            )
            existing = st.session_state["extra_sources"]
            st.session_state["extra_sources"] = pd.concat([existing, row], ignore_index=True)
            st.success("New source added to knowledge base")


def student_toolkit_tab(user: str) -> None:
    st.subheader("Student Toolkit")

    st.markdown("### What-if Simulator")
    c1, c2, c3 = st.columns(3)
    with c1:
        cur_att = st.number_input("Current attendance %", 0.0, 100.0, 78.0, 0.1, key="tool_attendance")
        done_classes = st.number_input("Classes completed", 1, 500, 60, key="tool_done_classes")
    with c2:
        upcoming = st.number_input("Upcoming classes", 0, 200, 20, key="tool_upcoming")
        attend_upcoming = st.number_input("Planned attended classes", 0, 200, 16, key="tool_attend_upcoming")
    with c3:
        cur_cgpa = st.number_input("Current CGPA", 0.0, 10.0, 7.2, 0.01, key="tool_cgpa")
        credits_done = st.number_input("Credits completed", 1, 250, 90, key="tool_credits_done")
    expected_gp = st.slider("Expected grade points in next credits", 0.0, 10.0, 8.0, 0.1, key="tool_expected_gp")
    future_credits = st.slider("Future credits", 1, 40, 20, key="tool_future_credits")

    proj_att = projected_attendance(cur_att, done_classes, upcoming, attend_upcoming)
    proj_cg = projected_cgpa(cur_cgpa, credits_done, future_credits, expected_gp)
    st.caption(f"Projected attendance: {proj_att}%")
    st.caption(f"Projected CGPA: {proj_cg}")
    if st.button("Run risk check", key="tool_risk_check"):
        if proj_att < 75:
            st.warning("Attendance risk detected (<75%).")
            append_row(
                ALERTS_FILE,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "user": user,
                    "alert_type": "attendance_risk",
                    "details": f"Projected attendance {proj_att}",
                },
            )
        else:
            st.success("No attendance risk detected in current projection.")

    st.markdown("### Checklist Planner")
    checklist = st.session_state["checklist"]
    for i, item in enumerate(checklist):
        cols = st.columns([3, 2, 1])
        cols[0].write(item["task"])
        cols[1].write(item["due"])
        item["done"] = cols[2].checkbox("Done", value=item["done"], key=f"todo_{i}")

    with st.form("new_todo"):
        task = st.text_input("New task", key="tool_new_task")
        due = st.date_input("Due date", value=date.today(), key="tool_due_date")
        if st.form_submit_button("Add task") and task.strip():
            checklist.append({"task": task.strip(), "due": str(due), "done": False})
            st.success("Task added")

    st.markdown("### Scholarship / Internship Matcher")
    domain = st.selectbox("Interest domain", ["AI", "Core", "Research", "Product"], key="tool_domain")
    opportunities = pd.DataFrame(
        [
            {"name": "National Scholarship Track", "min_cgpa": 8.0, "domain": "Research", "deadline": "2026-04-15"},
            {"name": "Industry Internship Pool", "min_cgpa": 7.0, "domain": "Product", "deadline": "2026-03-20"},
            {"name": "AI Fellowship", "min_cgpa": 7.5, "domain": "AI", "deadline": "2026-05-10"},
        ]
    )
    opportunities["fit_score"] = opportunities.apply(
        lambda r: (60 if cur_cgpa >= r["min_cgpa"] else 20) + (40 if r["domain"] == domain else 15), axis=1
    )
    st.dataframe(opportunities.sort_values("fit_score", ascending=False), use_container_width=True)


def analytics_tab() -> None:
    st.subheader("Analytics and Quality")
    logs = pd.read_csv(QUERY_LOG_FILE)
    feedback = pd.read_csv(FEEDBACK_FILE)
    tickets = pd.read_csv(TICKETS_FILE)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total queries", len(logs))
    m2.metric("Avg confidence", round(logs["confidence"].mean(), 2) if not logs.empty else 0)
    m3.metric("Avg latency (ms)", int(logs["latency_ms"].mean()) if not logs.empty else 0)
    if feedback.empty:
        m4.metric("Helpful rate", "0%")
    else:
        m4.metric("Helpful rate", f"{(feedback['feedback'].eq('Helpful').mean() * 100):.1f}%")

    st.markdown("### Ticket Queue")
    if tickets.empty:
        st.caption("No tickets created.")
    else:
        st.dataframe(tickets.tail(20), use_container_width=True)

    st.markdown("### Evaluation (quick)")
    if st.button("Run mini evaluation", key="analytics_eval_btn"):
        kb_df = active_kb()
        eval_rows = [
            ("minimum attendance requirement", "attendance"),
            ("how to apply for internship", "internship"),
            ("grade revaluation process", "exam"),
        ]
        outcomes = []
        correct = 0
        for q, expected in eval_rows:
            out = retrieve_dataset_answer(q, kb_df, st.session_state["selected_university"], "All", 3)
            passed = out["category"] == expected
            correct += int(passed)
            outcomes.append(
                {
                    "query": q,
                    "expected": expected,
                    "predicted": out["category"],
                    "confidence": round(out["confidence"], 3),
                    "pass": passed,
                }
            )
        st.metric("Evaluation score", f"{(correct / len(eval_rows)) * 100:.1f}%")
        st.dataframe(pd.DataFrame(outcomes), use_container_width=True)


def feedback_and_ratings_tab() -> None:
    st.subheader("Feedback and Ratings")
    feedback_df = pd.read_csv(FEEDBACK_FILE)
    if "rating" not in feedback_df.columns:
        feedback_df["rating"] = ""

    rating_series = pd.to_numeric(feedback_df["rating"], errors="coerce").dropna()
    avg_rating = float(rating_series.mean()) if not rating_series.empty else 0.0
    total_ratings = int(rating_series.count())
    st.metric("Global Rating", f"{avg_rating:.1f}/5", total_ratings)
    st.caption("Ratings are visible to all users.")

    with st.form("global_feedback_form"):
        user = st.text_input("Your username (optional)", value=st.session_state.get("active_user", "guest"))
        role = st.selectbox("Role", ["guest", "student", "teacher", "parent", "developer_admin"], key="global_role")
        feedback = st.radio("Feedback type", ["Helpful", "Not helpful"], horizontal=True)
        rating = st.slider("Rating", 1, 5, 4)
        comment = st.text_area("Comments / Suggestions")
        submit = st.form_submit_button("Submit feedback")
        if submit:
            append_row(
                FEEDBACK_FILE,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "user": user or "guest",
                    "role": role,
                    "university": st.session_state["selected_university"],
                    "query": "",
                    "response": "",
                    "confidence": "",
                    "feedback": feedback,
                    "rating": rating,
                    "comment": comment,
                },
            )
            st.success("Feedback submitted")

    latest_feedback = pd.read_csv(FEEDBACK_FILE)
    st.markdown("### Feedback Data")
    st.dataframe(latest_feedback.tail(50), use_container_width=True)
    st.download_button(
        "Download feedback data (CSV)",
        data=latest_feedback.to_csv(index=False).encode("utf-8"),
        file_name="feedback_export.csv",
        mime="text/csv",
    )


def parent_tab() -> None:
    st.subheader("Parent Overview")
    st.info("Read-only dashboard for progress tracking and official process visibility.")

    col1, col2, col3 = st.columns(3)
    col1.markdown("<div class='metric-card'><b>Attendance Risk</b><br>Monitor weekly</div>", unsafe_allow_html=True)
    col2.markdown("<div class='metric-card'><b>Exam Cycle</b><br>Follow notices</div>", unsafe_allow_html=True)
    col3.markdown("<div class='metric-card'><b>Internship Stage</b><br>Preparation phase</div>", unsafe_allow_html=True)

    st.markdown("### Latest conversation snapshot")
    if st.session_state["conversation"]:
        last = st.session_state["conversation"][-1]
        st.write(f"Q: {last['q']}")
        st.write(f"A: {last['a']}")
    else:
        st.caption("No conversation yet.")

    st.markdown("### Recent Alerts")
    alerts = pd.read_csv(ALERTS_FILE)
    if alerts.empty:
        st.caption("No alerts yet.")
    else:
        st.dataframe(alerts.tail(10), use_container_width=True)


def admin_tab() -> None:
    st.subheader("Developer / Admin Console")
    st.caption("Manage review queue, add data sources, and monitor quality.")

    st.markdown("### Review queue")
    queue = st.session_state["review_queue"]
    if not queue:
        st.caption("No pending low-confidence queries.")
    else:
        for i, item in enumerate(queue):
            with st.expander(f"{i+1}. {item['query']} ({item['confidence']:.2f})"):
                approved = st.text_area("Approved answer", value=item["suggested_answer"], key=f"approved_{i}")
                if st.button("Approve and add", key=f"approve_btn_{i}"):
                    row = pd.DataFrame(
                        [
                            {
                                "question": item["query"],
                                "answer": approved,
                                "university": st.session_state["selected_university"],
                                "category": infer_category(item["query"]),
                                "source": "admin_review",
                                "last_updated": str(date.today()),
                                "policy_link": "https://www.ugc.gov.in/",
                            }
                        ]
                    )
                    st.session_state["extra_sources"] = pd.concat([st.session_state["extra_sources"], row], ignore_index=True)
                    queue.pop(i)
                    st.success("Approved answer added as source")
                    st.rerun()

    st.markdown("### Upload extra source CSV")
    uploaded = st.file_uploader("CSV with question and answer columns", type=["csv"], key="admin_source_upload")
    if uploaded is not None:
        add_df = pd.read_csv(uploaded)
        if "question" in add_df.columns and "answer" in add_df.columns:
            add_df = map_university_labels(add_df)
            if "category" not in add_df.columns:
                add_df["category"] = add_df["question"].astype(str).apply(infer_category)
            if "source" not in add_df.columns:
                add_df["source"] = "admin_upload"
            if "last_updated" not in add_df.columns:
                add_df["last_updated"] = str(date.today())
            if "policy_link" not in add_df.columns:
                add_df["policy_link"] = "https://www.ugc.gov.in/"
            st.session_state["extra_sources"] = pd.concat([st.session_state["extra_sources"], add_df], ignore_index=True)
            st.success(f"Added {len(add_df)} rows into active sources")
        else:
            st.error("CSV must include question and answer columns")

    st.markdown("### Policy Change Detector")
    policy_file = st.file_uploader("Upload policy text/CSV for change tracking", type=["txt", "md", "csv"], key="policy_detector")
    if policy_file is not None:
        file_bytes = policy_file.getvalue()
        digest = hashlib.sha256(file_bytes).hexdigest()
        prev = st.session_state["policy_hashes"].get(policy_file.name)
        if prev is None:
            st.success("New policy file registered.")
        elif prev != digest:
            st.warning("Policy change detected from previous upload.")
        else:
            st.info("No policy change detected.")
        st.session_state["policy_hashes"][policy_file.name] = digest

    st.markdown("### Quality snapshot")
    logs = pd.read_csv(QUERY_LOG_FILE)
    if logs.empty:
        st.caption("No queries logged yet")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total queries", len(logs))
        c2.metric("Avg confidence", round(logs["confidence"].mean(), 2))
        c3.metric("Escalation rate", f"{(logs['escalated'].mean()*100):.1f}%")
        st.dataframe(logs.tail(20), use_container_width=True)


def main() -> None:
    ensure_storage()
    init_session()

    st.markdown(
        """
<div class='main-banner'>
  <div class='main-title'>🎓 UniAssist India</div>
  <div class='main-sub'>Dataset-grounded academic and internship assistant for Indian higher education workflows.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    render_sidebar()
    kb_df = active_kb()

    student_t, teacher_t, parent_t, admin_t, feedback_t = st.tabs(
        ["Student", "Teacher", "Parent", "Developer Admin", "Feedback & Ratings"]
    )

    with student_t:
        ok, user = role_login_gate("student", "Student")
        if ok:
            assistant_tab(kb_df, key_prefix="student", user=user, role="student")
            st.divider()
            student_toolkit_tab(user=user)

    with teacher_t:
        ok, user = role_login_gate("teacher", "Teacher")
        if ok:
            teacher_tab(user=user)

    with parent_t:
        ok, _ = role_login_gate("parent", "Parent")
        if ok:
            parent_tab()

    with admin_t:
        ok, user = role_login_gate("developer_admin", "Developer Admin")
        if ok:
            assistant_tab(kb_df, key_prefix="admin", user=user, role="developer_admin")
            st.divider()
            admin_tab()
            st.divider()
            analytics_tab()

    with feedback_t:
        feedback_and_ratings_tab()

    st.divider()
    st.caption("Author: Mehul Kumar")
    st.caption("© 2026 UniAssist India | Dataset-first retrieval system with role-based workflows")


if __name__ == "__main__":
    main()
