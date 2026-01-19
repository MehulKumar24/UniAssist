
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

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 700;
    color: #1f4ed8;
    text-align: center;
}
.sub-title {
    font-size: 18px;
    color: #555;
    text-align: center;
    margin-bottom: 30px;
}
.answer-box {
    background-color: #f0f7ff;
    padding: 15px;
    border-radius: 8px;
    border-left: 6px solid #1f4ed8;
}
.footer {
    font-size: 12px;
    color: #777;
    text-align: center;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🎓 UniAssist</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Academic & Internship Guidance Assistant</div>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("UniAssist_training_data.csv")

qa_frame = load_data()
questions = qa_frame["question"].astype(str).tolist()
answers = qa_frame["answer"].astype(str).tolist()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
question_embeddings = model.encode(questions)

# ---------------- SAFETY SETTINGS ----------------
SIMILARITY_THRESHOLD = 0.65

SAFE_FALLBACK_MESSAGE = (
    "I’m sorry, I don’t have reliable information on this topic. "
    "UniAssist currently handles academic and internship-related queries only."
)

# ---------------- RETRIEVAL FUNCTION ----------------
def get_safe_answer(user_query):
    query_vec = model.encode([user_query])
    scores = cosine_similarity(query_vec, question_embeddings)[0]

    best_index = scores.argmax()
    best_score = scores[best_index]

    if best_score < SIMILARITY_THRESHOLD:
        return SAFE_FALLBACK_MESSAGE

    return answers[best_index]

# ---------------- INPUT SECTION ----------------
st.subheader("Ask your question")

user_query = st.text_input(
    "Enter your query below",
    placeholder="e.g., What is the minimum attendance requirement?"
)

ask_button = st.button("Get Answer")

# ---------------- RESPONSE SECTION ----------------
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

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("About UniAssist")
    st.write("""
    UniAssist is an academic-focused assistant designed to help
    students with university rules, attendance policies, internships,
    examinations, and grading-related queries.
    """)

    st.subheader("Scope")
    st.write("""
    • University policies  
    • Academic regulations  
    • Internship guidance  
    • Examination rules  
    """)

    st.subheader("Disclaimer")
    st.write("""
    This tool provides informational guidance only.
    For official decisions, always refer to university notifications.
    """)

    st.subheader("Author")
    st.write("Mehul Kumar")

# ---------------- FOOTER ----------------
st.markdown(
    "<div class='footer'>© 2026 UniAssist | Academic Project & Research Prototype</div>",
    unsafe_allow_html=True
)
