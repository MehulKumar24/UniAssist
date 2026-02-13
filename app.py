import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import html

st.set_page_config(page_title="UniAssist", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# ---------------- Session state defaults ----------------
defaults = {
    'search_history': [], 'bookmarks': [], 'feedback_data': [], 'admin_mode': False,
    'admin_password': "admin123", 'custom_qa_pairs': [], 'rate_limit_count': 0,
    'faq_page': 0, 'fb_q': '', 'fb_c': '', 'fb_r': 3, 'ex_query': '', 'new_q': '', 'new_a': '', 'new_cat_name': ''
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------- Enhanced CSS (UI) ----------------
st.markdown(
    """<style>
    body{background:#ffffff;color:#000;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif}
    .main-title{font-size:48px;font-weight:800;color:#1f4ed8;text-align:center;margin-bottom:5px;letter-spacing:-0.5px}
    .sub-title{font-size:16px;color:#333;text-align:center;margin-bottom:25px;font-weight:500}
    .answer-box{background:linear-gradient(135deg, #e6f4ff 0%, #f0f8ff 100%);padding:22px;
    border-radius:12px;border-left:5px solid #1f4ed8;color:#000;margin:15px 0;box-shadow:0 2px 8px rgba(31,78,216,0.08)}
    .confidence-high{color:#059669;font-weight:700;font-size:16px}
    .confidence-medium{color:#d97706;font-weight:700;font-size:16px}
    .confidence-low{color:#dc2626;font-weight:700;font-size:16px}
    .footer{font-size:12px;color:#666;text-align:center;margin-top:40px;padding-top:20px;border-top:1px solid #e5e7eb}
    .stat-card{background:#f9fafb;padding:16px;border-radius:10px;border-left:3px solid #1f4ed8;color:#000}
    .sidebar-section{background:#f0f4f9;padding:14px;border-radius:8px;margin:10px 0;color:#000}
    </style>""",
    unsafe_allow_html=True
)

# ---------------- Data persistence (safe write) ----------------
def save_data():
    try:
        temp_file = 'uniassist_data_tmp.json'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump({
                'bookmarks': st.session_state.bookmarks,
                'feedback': st.session_state.feedback_data,
                'custom_qa': st.session_state.custom_qa_pairs
            }, f, ensure_ascii=False, indent=2)
        os.replace(temp_file, 'uniassist_data.json')
        return True
    except Exception as e:
        st.error(f"❌ Save error: {e}")
        return False

def load_data_from_file():
    if os.path.exists('uniassist_data.json'):
        try:
            with open('uniassist_data.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                st.session_state.bookmarks = data.get('bookmarks', [])
                st.session_state.feedback_data = data.get('feedback', [])
                st.session_state.custom_qa_pairs = data.get('custom_qa', [])
        except Exception:
            # If file is corrupted or unreadable, keep current session defaults (do not crash)
            st.session_state.bookmarks = st.session_state.get('bookmarks', [])
            st.session_state.feedback_data = st.session_state.get('feedback_data', [])
            st.session_state.custom_qa_pairs = st.session_state.get('custom_qa_pairs', [])

load_data_from_file()

# ---------------- Load Q&A CSV ----------------
@st.cache_data
def load_qa():
    try:
        df = pd.read_csv("UniAssist_training_data.csv")
        # ensure required columns exist; fallback to empty lists
        required = ["question", "answer", "category_name"]
        if not all(col in df.columns for col in required):
            return [], [], []
        return df["question"].fillna("").tolist(), df["answer"].fillna("").tolist(), df["category_name"].fillna("").tolist()
    except Exception:
        return [], [], []

q, a, c = load_qa()
# append any custom QA pairs saved in session state
if st.session_state.custom_qa_pairs:
    for qa in st.session_state.custom_qa_pairs:
        q.append(qa.get('question', ''))
        a.append(qa.get('answer', ''))
        c.append(qa.get('category', 'Custom'))

# ---------------- Load embedding model ----------------
@st.cache_resource
def load_model_embed():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

model = load_model_embed()
if model is None:
    # show a non-blocking message; search will gracefully handle None model
    st.warning("Embedding model failed to load — semantic search will be unavailable.")

# ---------------- Build embeddings (cached) ----------------
@st.cache_resource
def build_embeddings(questions_tuple):
    """
    Accepts a tuple of questions (strings). Uses the outer-scope model.
    This avoids passing the model object into the cache key (which is unhashable).
    """
    if model is None or not questions_tuple:
        return np.array([])
    try:
        questions_list = list(questions_tuple)
        embs = model.encode(questions_list, show_progress_bar=False)
        return np.asarray(embs).astype("float32")
    except Exception:
        return np.array([])

# Use tuple(q) to make the cache key hashable
embeddings = build_embeddings(tuple(q))
categories = sorted(set(cat for cat in c if str(cat).strip()))

THRESHOLD = 0.50

# ---------------- Search function ----------------
def search(query):
    # reset rate-limit counter (prevent permanent lock)
    if st.session_state.rate_limit_count > 100:
        st.session_state.rate_limit_count = 0

    if model is None or embeddings.size == 0 or len(q) == 0:
        return None, 0, []

    # count valid searches only
    st.session_state.rate_limit_count += 1
    try:
        query_vec = model.encode([query], show_progress_bar=False)
        scores = cosine_similarity(query_vec, embeddings)[0]
        top_idx = int(np.argmax(scores))
        if float(scores[top_idx]) < THRESHOLD:
            return None, 0, []
        related = []
        # safer related collection: skip top_idx, collect up to 3 with threshold tolerance
        for idx in np.argsort(scores)[::-1]:
            if int(idx) == int(top_idx):
                continue
            if len(related) >= 3:
                break
            if float(scores[int(idx)]) >= (THRESHOLD - 0.05):
                related.append({'q': q[int(idx)], 'a': a[int(idx)], 's': float(scores[int(idx)])})
        return a[top_idx], float(scores[top_idx]), related
    except Exception:
        return None, 0, []

# ---------------- Bookmarks ----------------
def add_bookmark(qn, ans):
    if not any(b['question'] == qn for b in st.session_state.bookmarks):
        st.session_state.bookmarks.append({'timestamp': datetime.now().isoformat(), 'question': qn, 'answer': ans})
        save_data()
        return True
    return False

def remove_bookmark(qn):
    st.session_state.bookmarks = [b for b in st.session_state.bookmarks if b['question'] != qn]
    save_data()

# ---------------- Feedback ----------------
def add_feedback(qry, rate, comm=""):
    try:
        st.session_state.feedback_data.append({
            'timestamp': datetime.now().isoformat(),
            'query': str(qry),
            'rating': int(rate),
            'comment': str(comm)
        })
        # bound feedback storage to avoid unbounded growth
        if len(st.session_state.feedback_data) > 500:
            st.session_state.feedback_data.pop(0)
        save_data()
    except Exception as e:
        st.error(f"Feedback save error: {e}")

# ---------------- Sidebar (UI) ----------------
with st.sidebar:
    st.markdown("<div style='text-align:center;margin-bottom:20px'><h2 style='color:#1f4ed8'>📚 UniAssist</h2></div>", unsafe_allow_html=True)

    nav = st.radio("📍 Choose a Section:", ["🏠 Home", "📚 Browse FAQ", "⭐ Bookmarks", "📝 Feedback", "🔐 Admin"], key="nav_menu")

    st.divider()

    # Quick Stats removed as requested (kept commented for future use)
    # st.markdown("<div class='sidebar-section'><b>📊 Quick Stats</b><br>", unsafe_allow_html=True)
    # ... (removed) ...
    # st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'><b>📖 Dataset Info</b><br>", unsafe_allow_html=True)
    st.write(f"• **Total Q&A:** {len(q)}")
    st.write(f"• **Categories:** {len(categories)}")
    st.write(f"• **Status:** ✅ Active")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📋 Using UniAssist"):
        st.markdown("""
        **Tips for best results:**
        
        1️⃣ **Be Specific** - Use detailed questions  
        2️⃣ **Use Keywords** - Include relevant terms  
        3️⃣ **Browse FAQ** - Explore by category  
        4️⃣ **Save Answers** - Bookmark important Q&As  
        5️⃣ **Rate Answers** - Help us improve
        """)

    with st.expander("ℹ️ About"):
        st.markdown("""
        **UniAssist v2.0**
        
        ✨ Semantic search  
        🎓 1075+ Q&A pairs  
        🔖 Save bookmarks  
        💬 Feedback system  
        🔐 Admin panel
        
        © 2026 UniAssist
        """)

# ---------------- Page header ----------------
st.markdown("<div class='main-title'>🎓 UniAssist</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-Powered Academic Guidance Assistant</div>", unsafe_allow_html=True)
st.divider()

# ---------------- HOME ----------------
if nav == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🤔 Ask Your Question")

        query = st.text_input("Enter your question:", placeholder="What is the minimum attendance requirement?", key="ex_query", label_visibility="collapsed")

        btn = st.button("🔍 Get Answer", use_container_width=True, type="primary")

        if btn and query.strip():
            query = query.strip()[:500]
            with st.spinner("🔄 Searching knowledge base..."):
                ans, conf, rel = search(query)

            if ans:
                # append search history (bounded to avoid memory growth)
                st.session_state.search_history.append({'query': query, 'confidence': conf, 'timestamp': datetime.now().isoformat()})
                if len(st.session_state.search_history) > 200:
                    st.session_state.search_history.pop(0)

                st.markdown(f"<div class='answer-box'>{html.escape(ans)}</div>", unsafe_allow_html=True)

                color = "confidence-high" if conf >= 0.7 else "confidence-medium" if conf >= 0.5 else "confidence-low"
                text = "🟢 High" if conf >= 0.7 else "🟡 Medium" if conf >= 0.5 else "🔴 Low"
                st.markdown(f"<p style='text-align: center;'><span class='{color}'>{text} Confidence: {conf:.0%}</span></p>", unsafe_allow_html=True)

                col_bm, col_cp, col_spc, col_spc2, col_rt = st.columns(5)
                unique_hash = abs(hash(query)) % (10**12)
                with col_bm:
                    if st.button("⭐ Bookmark", use_container_width=True, key=f"bm_{unique_hash}"):
                        if add_bookmark(query, ans):
                            st.success("✅ Saved to bookmarks!")
                        else:
                            st.info("✓ Already bookmarked")
                with col_cp:
                    if st.button("📋 Copy", use_container_width=True, key=f"cp_{unique_hash}"):
                        st.info("✓ Use browser copy")
                with col_rt:
                    if st.button("👍 Rate", use_container_width=True, key=f"rt_{unique_hash}"):
                        st.info("→ Go to Feedback tab")

                if rel:
                    st.markdown("### 🔗 Related Questions")
                    for i, r in enumerate(rel, 1):
                        # use index-based key to avoid duplicates
                        with st.expander(f"Q{i}: {r['q'][:60]}... ({int(r['s']*100)}%)"):
                            st.write(r['a'])
                            if st.button(f"⭐ Save Q{i}", key=f"save_rel_{i}_{unique_hash}", use_container_width=True):
                                if add_bookmark(r['q'], r['a']):
                                    st.success("✅ Saved!")
                                else:
                                    st.info("✓ Already bookmarked")
            else:
                st.warning("⚠️ No reliable answer found. Try browsing FAQ or provide more details.", icon="⚠️")
        elif btn:
            st.error("❌ Please enter a question first.")

    with col2:
        st.markdown("### 📊 Your Activity")
        with st.container():
            st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
            st.metric("🔍 Searches", len(st.session_state.search_history))
            st.metric("⭐ Bookmarks", len(st.session_state.bookmarks))
            st.metric("💬 Ratings", len(st.session_state.feedback_data))
            if st.session_state.search_history:
                avg_conf = np.mean([h.get('confidence', 0) for h in st.session_state.search_history])
                st.metric("Avg Confidence", f"{avg_conf:.0%}")
            st.markdown("</div>", unsafe_allow_html=True)

# ---------------- BROWSE FAQ ----------------
elif nav == "📚 Browse FAQ":
    st.subheader("📚 Frequently Asked Questions")

    col1, col2 = st.columns([2, 1])
    with col1:
        cat = st.selectbox("📂 Filter by Category:", ["All Categories"] + categories, key="faq_cat")
    with col2:
        per_page = st.selectbox("📄 Per Page:", [5, 10, 20, 50], index=1, key="faq_per")

    # safer session-state check
    if '_faq_cat' not in st.session_state:
        st.session_state._faq_cat = cat
        st.session_state._faq_per = per_page
    if cat != st.session_state._faq_cat or per_page != st.session_state._faq_per:
        st.session_state.faq_page = 0
        st.session_state._faq_cat, st.session_state._faq_per = cat, per_page

    idx = list(range(len(q))) if cat == "All Categories" else [i for i, x in enumerate(c) if x == cat]
    pages = max(1, (len(idx) + per_page - 1) // per_page)
    if st.session_state.faq_page >= pages:
        st.session_state.faq_page = 0

    col_p, col_pg, col_n = st.columns([1, 2, 1])
    with col_p:
        if st.button("◀ Previous", use_container_width=True, key="faqp"):
            if st.session_state.faq_page > 0:
                st.session_state.faq_page -= 1
                st.rerun()
    with col_pg:
        st.markdown(f"<p style='text-align:center;font-weight:600'>Page {st.session_state.faq_page + 1} of {pages}</p>", unsafe_allow_html=True)
    with col_n:
        if st.button("Next ▶", use_container_width=True, key="faqn"):
            if st.session_state.faq_page < pages - 1:
                st.session_state.faq_page += 1
                st.rerun()

    st.divider()
    start = st.session_state.faq_page * per_page
    end = min(start + per_page, len(idx))

    if len(idx) > 0:
        for pos, i in enumerate(idx[start:end], 1):
            col_q, col_bm = st.columns([5, 1])
            with col_q:
                with st.expander(f"{pos}. {q[i][:75]}"):
                    st.write(a[i])
                    st.caption(f"📂 Category: {c[i]}")
            with col_bm:
                if st.button("⭐", key=f"faq_bm_{i}", help="Bookmark this Q&A"):
                    if add_bookmark(q[i], a[i]):
                        st.success("✅ Bookmarked!")
                    else:
                        st.info("✓ Already bookmarked")
    else:
        st.info("📭 No questions in this category")

# ---------------- BOOKMARKS ----------------
elif nav == "⭐ Bookmarks":
    st.subheader("⭐ My Saved Questions")

    if st.session_state.bookmarks:
        cols_display = st.columns([1, 4])
        with cols_display[0]:
            st.write(f"**Total:** {len(st.session_state.bookmarks)}")

        for i, bm in enumerate(st.session_state.bookmarks):
            col1, col2 = st.columns([5, 1])
            with col1:
                with st.expander(f"📌 {bm['question'][:75]}"):
                    st.write(bm['answer'])
                    st.caption(f"Saved: {bm['timestamp'][:10]}")
            with col2:
                if st.button("🗑️", key=f"rm_bm_{i}", help="Delete bookmark"):
                    remove_bookmark(bm['question'])
                    st.rerun()
    else:
        st.info("📌 No bookmarks yet. Save questions from Home or Browse FAQ!")

# ---------------- FEEDBACK ----------------
elif nav == "📝 Feedback":
    st.subheader("📝 Provide Feedback")

    col1, col2 = st.columns([2, 1])
    with col1:
        fb_q = st.text_input("Question (optional):", value=st.session_state.fb_q, key="feedback_q_input", label_visibility="collapsed", placeholder="Which question are you rating?")
        fb_c = st.text_area("Your Feedback:", value=st.session_state.fb_c, key="feedback_c_input", label_visibility="collapsed", height=120, placeholder="Share your experience...")
    with col2:
        st.write("**Rate this answer:**")
        fb_r = st.radio("Rating:", [1, 2, 3, 4, 5], value=st.session_state.fb_r, key="feedback_r_input", label_visibility="collapsed", horizontal=False)

    col_submit, col_clear = st.columns([3, 1])
    with col_submit:
        if st.button("✅ Submit Feedback", type="primary", use_container_width=True):
            if fb_c.strip():
                add_feedback(fb_q, fb_r, fb_c)
                st.success("✅ Thank you for your feedback!")
                st.session_state.fb_q = ""
                st.session_state.fb_c = ""
                st.session_state.fb_r = 3
                st.rerun()
            else:
                st.error("❌ Please add your feedback before submitting")

    with col_clear:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.fb_q = ""
            st.session_state.fb_c = ""
            st.session_state.fb_r = 3
            st.rerun()

    if st.session_state.feedback_data:
        st.divider()
        st.markdown("### 📊 Feedback Summary")
        rates = [f['rating'] for f in st.session_state.feedback_data if isinstance(f.get('rating'), int)]
        if rates:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Rating", f"{np.mean(rates):.1f}/5.0")
            with col2:
                st.metric("Total Responses", len(rates))
            with col3:
                st.metric("Latest", rates[-1] if rates else "N/A")

# ---------------- ADMIN ----------------
elif nav == "🔐 Admin":
    st.subheader("🔐 Admin Panel")

    if not st.session_state.admin_mode:
        st.warning("🔒 Password required to access admin panel")
        pwd = st.text_input("Enter Admin Password:", type="password", key="admin_pwd_input")
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🔓 Login", use_container_width=True, type="primary"):
                if pwd == st.session_state.admin_password:
                    st.session_state.admin_mode = True
                    st.rerun()
                else:
                    st.error("❌ Wrong password!")
    else:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.success("✅ Admin Mode Active", icon="✅")
        with col2:
            if st.button("🔒 Logout", use_container_width=True):
                st.session_state.admin_mode = False
                st.rerun()

        st.divider()
        t1, t2, t3, t4 = st.tabs(["➕ Add Q&A", "📋 Manage", "💬 Feedback", "📤 Export"])

        with t1:
            st.markdown("### Add New Q&A Pair")
            nq = st.text_input("Question:", value=st.session_state.new_q, key="admin_q_input", label_visibility="collapsed", placeholder="Enter question")
            na = st.text_area("Answer:", value=st.session_state.new_a, key="admin_a_input", label_visibility="collapsed", placeholder="Enter answer", height=100)

            cat_ch = st.radio("Add to:", ["Existing Category", "New Category"], key="cat_ch", horizontal=True)
            if cat_ch == "Existing Category":
                nc = st.selectbox("Select Category:", categories, key="Admin_cat")
            else:
                nc = st.text_input("New Category Name:", value=st.session_state.new_cat_name, key="admin_cat_input", label_visibility="collapsed")

            if st.button("➕ Add Q&A", type="primary", use_container_width=True):
                if nq.strip() and na.strip() and nc.strip():
                    st.session_state.custom_qa_pairs.append({
                        'question': nq.strip(), 'answer': na.strip(), 'category': nc.strip(), 'added_on': datetime.now().isoformat()
                    })
                    save_data()
                    # rebuild embeddings so new Q&A become searchable
                    try:
                        st.cache_resource.clear()
                    except Exception:
                        pass
                    st.success("✅ Q&A Added Successfully! (Reload page to index)")
                    st.session_state.new_q = ""
                    st.session_state.new_a = ""
                    st.session_state.new_cat_name = ""
                    st.rerun()
                else:
                    st.error("❌ All fields are required")

        with t2:
            st.markdown("### Manage Custom Q&A")
            if st.session_state.custom_qa_pairs:
                st.write(f"**Total Custom Q&A:** {len(st.session_state.custom_qa_pairs)}")
                # iterate over a copy to avoid index skipping on pop
                for i, qa in enumerate(st.session_state.custom_qa_pairs.copy()):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        with st.expander(f"Q: {qa['question'][:70]}..."):
                            st.write(f"**Answer:** {qa['answer']}")
                            st.caption(f"Category: {qa['category']} | Added: {qa['added_on'][:10]}")
                    with col2:
                        if st.button("🗑️", key=f"del_{i}", help="Delete this Q&A"):
                            # remove by matching question (safer than pop by index)
                            st.session_state.custom_qa_pairs = [x for x in st.session_state.custom_qa_pairs if x.get('question') != qa.get('question')]
                            save_data()
                            st.rerun()
            else:
                st.info("No custom Q&A pairs yet")

        with t3:
            st.markdown("### Feedback Analysis")
            if st.session_state.feedback_data:
                df_fb = pd.DataFrame(st.session_state.feedback_data)
                cols_show = [col for col in df_fb.columns if col in ['query', 'rating', 'comment', 'timestamp']]
                try:
                    st.dataframe(df_fb[cols_show] if cols_show else df_fb, use_container_width=True)
                except Exception:
                    pass

                rates = [f['rating'] for f in st.session_state.feedback_data if isinstance(f.get('rating'), int)]
                if rates:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Rating", f"{np.mean(rates):.1f}/5")
                    with col2:
                        st.metric("Total", len(rates))
                    with col3:
                        st.metric("Highest", max(rates))

                if st.button("🗑️ Clear All Feedback", key="clear_fb"):
                    st.session_state.feedback_data = []
                    save_data()
                    st.success("✅ Feedback cleared")
                    st.rerun()
            else:
                st.info("No feedback yet")

        with t4:
            st.markdown("### Export Data")
            st.info(f"📊 Bookmarks: {len(st.session_state.bookmarks)} | 💬 Feedback: {len(st.session_state.feedback_data)} | ❓ Q&A: {len(st.session_state.custom_qa_pairs)}")

            fmt = st.selectbox("Export Format:", ["JSON", "CSV"], key="exp_fmt")
            data = {
                'bookmarks': st.session_state.bookmarks,
                'feedback': st.session_state.feedback_data,
                'custom_qa': st.session_state.custom_qa_pairs,
                'exported': datetime.now().isoformat()
            }

            if fmt == "JSON":
                st.download_button("⬇️ Download JSON", json.dumps(data, indent=2), "uniassist_data.json", "application/json", use_container_width=True)
            else:
                if st.session_state.feedback_data:
                    csv = pd.DataFrame(st.session_state.feedback_data).to_csv(index=False)
                    st.download_button("⬇️ Download CSV", csv, "uniassist_feedback.csv", "text/csv", use_container_width=True)
                else:
                    st.info("No feedback data to export")

# ---------------- Footer ----------------
st.divider()
st.markdown("<div class='footer'>© 2026 UniAssist | AI-Powered Academic Guidance Assistant</div>", unsafe_allow_html=True)

