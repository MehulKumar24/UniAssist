
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="UniAssist",
    page_icon="🎓",
    layout="centered"
)

# =========================================================
# CUSTOM CSS — FIXED CONTRAST & COLORS
# =========================================================
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Main title */
    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #0f172a; /* dark navy */
        text-align: center;
        margin-bottom: 4px;
    }

    /* Subtitle */
    .sub-title {
        font-size: 18px;
        color: #334155; /* slate */
        text-align: center;
        margin-bottom: 30px;
    }

    /* Labels & headings */
    label, h3, h2, h1 {
        color: #0f172a !important;
    }

    /* Text input */
    input[type="text"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 8px !important;
        border: 1px solid #cbd5e1 !important;
    }

    /* Placeholder text */
    input::placeholder {
        color: #64748b !important;
    }

    /* Answer box */
    .answer-box {
        background-color: #ffffff;
        color: #0f172a;
        padding: 16px;
        border-radius: 10px;
        border-left: 6px solid #2563eb;
        margin-top: 15px;
        font-size: 16px;
        line-height: 1.6;
    }

    /* Buttons */
    button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600;
        border: none !important;
    }

    button:hover {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* Footer */
    .footer {
        font-size: 12px;
        color: #64748b;
        text-align: center;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HEADER
# =========================================================
st.markdown("<div class='main-title'>🎓 UniAssist</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Academic & Internship Guidance Assistant</div>",
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv("UniAssist_training_data.csv")

qa_frame = load_data()
questions = qa_frame["question"].astype(str).tolist()
answers = qa_frame["answer"].astype(str).tolist()

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
question_embeddings = model.encode(questions)

# =========================================================
# SAFETY SETTINGS
# =========================================================
SIMILARITY_THRESHOLD = 0.65

SAFE_FALLBACK_MESSAGE = (
    "I’m sorry, I don’t have reliable information on this topic. "
    "UniAssist currently handles academic and internship-related queries only."
)

# =========================================================
# RETRIEVAL FUNCTION
# =========================================================
def get_safe_answer(user_query: str) -> str:
    query_vec = model.encode([user_query])
    scores = cosine_similarity(query_vec, question_embeddings)[0]

    best_index = scores.argmax()
    best_score = scores[best_index]

    if best_score < SIMILARITY_THRESHOLD:
        return SAFE_FALLBACK_MESSAGE

    return answers[best_index]

# =========================================================
# INPUT SECTION
# =========================================================
st.subheader("Ask your question")

user_query = st.text_input(
    "Enter your query below",
    placeholder="e.g., What is the minimum attendance requirement?"
)

ask_button = st.button("Get Answer")

# =========================================================
# RESPONSE SECTION
# =========================================================
if ask_button:
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        response = get_safe_answer(user_query)
        st.markdown("### 📘 Answer")
        st.markdown(
            f"<div class='answer-box'>{response}</div>",
            unsafe_allow_html=True
        )

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("About UniAssist")
    st.write(
        """
        UniAssist is an academic-focused assistant designed to help students with:
        - University regulations
        - Attendance policies
        - Internship guidance
        - Examination and grading rules
        """
    )

    st.subheader("Scope")
    st.write(
        """
        • Academic regulations  
        • Internship eligibility & rules  
        • Examination policies  
        • Grading systems  
        """
    )

    st.subheader("Disclaimer")
    st.write(
        """
        This tool provides informational guidance only.
        For official decisions, always refer to university notifications.
        """
    )

    st.subheader("Author")
    st.write("Mehul Kumar")

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    "<div class='footer'>© 2026 UniAssist | Academic Project & Research Prototype</div>",
    unsafe_allow_html=True
)
