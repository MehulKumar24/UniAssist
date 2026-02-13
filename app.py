import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import html

st.set_page_config(page_title="UniAssist", page_icon="🎓", layout="wide")

# ---------------- SESSION STATE ----------------
defaults = {
    "search_history": [],
    "bookmarks": [],
    "feedback_data": [],
    "admin_mode": False,
    "admin_password": "admin123",
    "custom_qa_pairs": [],
    "faq_page": 0
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- DATA SAVE / LOAD ----------------
def save_data():
    try:
        temp_file = "uniassist_temp.json"
        with open(temp_file, "w") as f:
            json.dump({
                "bookmarks": st.session_state.bookmarks,
                "feedback": st.session_state.feedback_data,
                "custom_qa": st.session_state.custom_qa_pairs
            }, f)
        os.replace(temp_file, "uniassist_data.json")
    except Exception as e:
        st.error(f"Save error: {e}")

def load_data():
    try:
        if os.path.exists("uniassist_data.json"):
            with open("uniassist_data.json", "r") as f:
                data = json.load(f)
            st.session_state.bookmarks = data.get("bookmarks", [])
            st.session_state.feedback_data = data.get("feedback", [])
            st.session_state.custom_qa_pairs = data.get("custom_qa", [])
    except:
        pass

load_data()

# ---------------- LOAD QA ----------------
@st.cache_data
def load_qa():
    try:
        df = pd.read_csv("UniAssist_training_data.csv")
        if not all(col in df.columns for col in ["question", "answer", "category_name"]):
            return [], [], []
        return df["question"].fillna("").tolist(), \
               df["answer"].fillna("").tolist(), \
               df["category_name"].fillna("General").tolist()
    except:
        return [], [], []

q, a, c = load_qa()

for qa in st.session_state.custom_qa_pairs:
    q.append(qa["question"])
    a.append(qa["answer"])
    c.append(qa["category"])

categories = sorted(set(cat for cat in c if str(cat).strip()))

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except:
        return None

model = load_model()

@st.cache_resource
def build_embeddings(model, questions):
    if model is None or not questions:
        return np.array([])
    return model.encode(questions, show_progress_bar=False).astype("float32")

embeddings = build_embeddings(model, tuple(q))

# ---------------- SEARCH ----------------
def search(query):
    if model is None or embeddings.size == 0:
        return None, 0, []

    query_vec = model.encode([query])
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_idx = np.argmax(scores)

    if scores[top_idx] < 0.5:
        return None, 0, []

    related = []
    for i in np.argsort(scores)[::-1]:
        if i == top_idx:
            continue
        if len(related) >= 3:
            break
        related.append((q[i], a[i], scores[i]))

    return a[top_idx], scores[top_idx], related

# ---------------- BOOKMARK ----------------
def add_bookmark(qn, ans):
    if not any(b["question"] == qn for b in st.session_state.bookmarks):
        st.session_state.bookmarks.append({
            "question": qn,
            "answer": ans,
            "timestamp": datetime.now().isoformat()
        })
        save_data()

# ---------------- FEEDBACK ----------------
def add_feedback(query, rating, comment):
    st.session_state.feedback_data.append({
        "query": query,
        "rating": int(rating),
        "comment": comment,
        "timestamp": datetime.now().isoformat()
    })
    if len(st.session_state.feedback_data) > 500:
        st.session_state.feedback_data.pop(0)
    save_data()

# ---------------- UI ----------------
st.title("🎓 UniAssist")

nav = st.sidebar.radio("Navigation", ["Home", "FAQ", "Bookmarks", "Feedback", "Admin"])

# ---------------- HOME ----------------
if nav == "Home":
    query = st.text_input("Ask question")

    if st.button("Search"):
        if query.strip():
            ans, conf, rel = search(query)

            if ans:
                st.markdown(f"**Answer:** {html.escape(ans)}")
                st.write(f"Confidence: {conf:.2f}")

                if st.button("⭐ Bookmark", key=f"bm_{query}"):
                    add_bookmark(query, ans)

                if rel:
                    st.subheader("Related")
                    for rq, ra, rs in rel:
                        with st.expander(rq):
                            st.write(ra)
            else:
                st.warning("No answer found")

# ---------------- FAQ ----------------
elif nav == "FAQ":
    cat = st.selectbox("Category", ["All"] + categories)
    idx = range(len(q)) if cat == "All" else [i for i, x in enumerate(c) if x == cat]

    for i in idx[:50]:
        with st.expander(q[i]):
            st.write(a[i])
            if st.button("⭐", key=f"faq_{i}"):
                add_bookmark(q[i], a[i])

# ---------------- BOOKMARK ----------------
elif nav == "Bookmarks":
    for i, bm in enumerate(st.session_state.bookmarks):
        with st.expander(bm["question"]):
            st.write(bm["answer"])

# ---------------- FEEDBACK ----------------
elif nav == "Feedback":
    with st.form("fb"):
        qy = st.text_input("Question")
        cm = st.text_area("Feedback")
        rt = st.slider("Rating", 1, 5, 3)
        sub = st.form_submit_button("Submit")

        if sub:
            if cm.strip():
                add_feedback(qy, rt, cm)
                st.success("Feedback submitted")
                st.rerun()
            else:
                st.error("Write feedback")

    if st.session_state.feedback_data:
        st.subheader("Feedback Summary")
        df = pd.DataFrame(st.session_state.feedback_data)
        st.dataframe(df, use_container_width=True)

# ---------------- ADMIN ----------------
elif nav == "Admin":
    pwd = st.text_input("Password", type="password")

    if pwd == st.session_state.admin_password:
        st.success("Admin mode")

        nq = st.text_input("Question")
        na = st.text_area("Answer")
        nc = st.text_input("Category")

        if st.button("Add Q&A"):
            if nq and na and nc:
                st.session_state.custom_qa_pairs.append({
                    "question": nq,
                    "answer": na,
                    "category": nc,
                    "timestamp": datetime.now().isoformat()
                })
                save_data()
                st.cache_resource.clear()
                st.success("Added. Reloading...")
                st.rerun()
