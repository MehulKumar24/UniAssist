import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="UniAssist",
    page_icon="🎓",
    layout="centered"
)

# ---------------- SAFE CUSTOM CSS ----------------
st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #f8fafc;
}

/* Headings */
h1, h2, h3 {
    color: #0f172a;
    font-family: "Segoe UI", sans-serif;
}

/* IMPORTANT: prevent label blocking clicks */
label {
    color: #64748b !important;
    pointer-events: none !important;
}

/* Text input box */
input[type="text"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-radius: 8px !important;
    border: 1px solid #cbd5e1 !important;
    padding: 10px !important;
    pointer-events: auto !important;
    opacity: 1 !important;
}

/* Button */
button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Answer card */
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

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0b1220;
    color: #e5e7eb;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p {
    color: #e5e7eb !important;
}

/* Footer */
.footer {
    font-size: 12px;
    color: #64748b;
    text-align: center;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🎓 UniAssist</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#475569;'>Academic & Internship Guidance Assistant</p>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("UniAssist_training_data.csv")

qa_df = load_data()
questions = qa_df["question"].astype(str).tolist()
answers = qa_df["answer"].astype(str).tolist()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
question_embeddings = model.encode(questions)

# ---------------- SETTINGS ----------------
SIMILARITY_THRESHOLD = 0.65

FALLBACK_MESSAGE = (
    "I’m sorry, I don’t have reliable information on this topic. "
    "UniAssist currently handles academic and internship-related queries only."
)

# ---------------- LOGIC ----------------
def get_answer(query):
    query_vec = model.encode([query])
    scores = cosine_similarity(query_vec, question_embeddings)[0]
    best_idx = scores.argmax()

    if scores[best_idx] < SIMILARITY_THRESHOLD:
        return FALLBACK_MESSAGE

    return answers[best_idx]

# ---------------- INPUT ----------------
st.subheader("Ask your question")

user_query = st.text_input(
    label="Enter your query",
    placeholder="e.g., What is the minimum attendance requirement?"
)

if st.button("Get Answer"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        response = get_answer(user_query)
        st.markdown("### 📘 Answer")
        st.markdown(
            f"<div class='answer-box'>{response}</div>",
            unsafe_allow_html=True
        )

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("About UniAssist")
    st.write("""
    UniAssist is an academic-focused assistant designed to help students with:
    • University regulations  
    • Attendance policies  
    • Internship guidance  
    • Examination and grading rules
    """)

    st.subheader("Disclaimer")
    st.write("""
    This tool provides informational guidance only.
    Always refer to official university notifications.
    """)

    st.subheader("Author")
    st.write("Mehul Kumar")

# ---------------- FOOTER ----------------
st.markdown(
    "<div class='footer'>© 2026 UniAssist | Academic Project & Research Prototype</div>",
    unsafe_allow_html=True
)
