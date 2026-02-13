# app.py - UniAssist (fixed, full)
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import html
import streamlit.components.v1 as components

st.set_page_config(page_title="UniAssist", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# ---------------- Session defaults ----------------
defaults = {
    'search_history': [], 'bookmarks': [], 'feedback_data': [], 'admin_mode': False,
    'admin_password': "admin123", 'custom_qa_pairs': [], 'rate_limit_count': 0,
    'faq_page': 0, 'fb_q': '', 'fb_c': '', 'fb_r': 3, 'ex_query': '', 'new_q': '', 'new_a': '', 'new_cat_name': '',
    'nav_menu': "🏠 Home"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- CSS ----------------
st.markdown("""<style>
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
</style>""", unsafe_allow_html=True)

# ---------------- Safe data persistence ----------------
DATA_FILE = 'uniassist_data.json'
def save_data():
    try:
        temp = DATA_FILE + ".tmp"
        with open(temp, 'w', encoding='utf-8') as f:
            json.dump({
                'bookmarks': st.session_state.bookmarks,
                'feedback': st.session_state.feedback_data,
                'custom_qa': st.session_state.custom_qa_pairs
            }, f, ensure_ascii=False, indent=2)
        os.replace(temp, DATA_FILE)
        return True
    except Exception as e:
        st.error(f"❌ Save error: {e}")
        return False

def load_data_from_file():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            st.session_state.bookmarks = data.get('bookmarks', [])
            st.session_state.feedback_data = data.get('feedback', [])
            st.session_state.custom_qa_pairs = data.get('custom_qa', [])
        except Exception:
            # keep defaults if corrupted
            st.session_state.bookmarks = st.session_state.get('bookmarks', [])
            st.session_state.feedback_data = st.session_state.get('feedback_data', [])
            st.session_state.custom_qa_pairs = st.session_state.get('custom_qa_pairs', [])

load_data_from_file()

# ---------------- Load Q&A ----------------
@st.cache_data
def load_qa():
    try:
        df = pd.read_csv("UniAssist_training_data.csv")
        req = ["question", "answer", "category_name"]
        if not all(col in df.columns for col in req):
            return [], [], []
        return df["question"].fillna("").tolist(), df["answer"].fillna("").tolist(), df["category_name"].fillna("").tolist()
    except Exception:
        return [], [], []

q, a, c = load_qa()
# append session custom QA at runtime (not cached)
if st.session_state.custom_qa_pairs:
    for qa in st.session_state.custom_qa_pairs:
        q.append(qa.get('question', ''))
        a.append(qa.get('answer', ''))
        c.append(qa.get('category', 'Custom'))

# ---------------- Model loading ----------------
@st.cache_resource
def load_model_embed():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

model = load_model_embed()
if model is None:
    st.warning("Embedding model not available. Semantic search will be disabled until model loads.")

# ---------------- Build embeddings (cache only on questions tuple) ----------------
@st.cache_resource
def build_embeddings(questions_tuple):
    if model is None or not questions_tuple:
        return np.array([])
    try:
        questions = list(questions_tuple)
        embs = model.encode(questions, show_progress_bar=False)
        return np.asarray(embs).astype("float32")
    except Exception:
        return np.array([])

embeddings = build_embeddings(tuple(q))
categories = sorted(set(x for x in c if str(x).strip()))

THRESHOLD = 0.50

# ---------------- Search ----------------
def search(query):
    if st.session_state.rate_limit_count > 100:
        st.session_state.rate_limit_count = 0
    if model is None or embeddings.size == 0 or len(q) == 0:
        return None, 0.0, []
    st.session_state.rate_limit_count += 1
    try:
        qvec = model.encode([query], show_progress_bar=False)
        scores = cosine_similarity(qvec, embeddings)[0]
        top_idx = int(np.argmax(scores))
        top_score = float(scores[top_idx])
        if top_score < THRESHOLD:
            return None, 0.0, []
        related = []
        for idx in np.argsort(scores)[::-1]:
            if int(idx) == top_idx:
                continue
            if len(related) >= 3:
                break
            s = float(scores[int(idx)])
            if s >= (THRESHOLD - 0.05):
                related.append({'q': q[int(idx)], 'a': a[int(idx)], 's': s})
        return a[top_idx], top_score, related
    except Exception:
        return None, 0.0, []

# ---------------- Bookmarks ----------------
def add_bookmark(qn, ans):
    if not any(b.get('question') == qn for b in st.session_state.bookmarks):
        st.session_state.bookmarks.append({'timestamp': datetime.now().isoformat(), 'question': qn, 'answer': ans})
        save_data()
        return True
    return False

def remove_bookmark(qn):
    st.session_state.bookmarks = [b for b in st.session_state.bookmarks if b.get('question') != qn]
    save_data()

# ---------------- Feedback ----------------
def add_feedback(qry, rate, comm):
    try:
        st.session_state.feedback_data.append({
            'timestamp': datetime.now().isoformat(),
            'query': str(qry),
            'rating': int(rate),
            'comment': str(comm)
        })
        if len(st.session_state.feedback_data) > 1000:
            st.session_state.feedback_data = st.session_state.feedback_data[-1000:]
        save_data()
        return True
    except Exception as e:
        st.error(f"Feedback save error: {e}")
        return False

# ---------------- small helper: copy-to-clipboard component ----------------
def copy_button(text, button_id):
    # simple JS snippet that copies the string to clipboard
    safe_text = html.escape(text).replace("\n", "\\n").replace("'", "\\'")
    component = f"""
    <button id="btn_{button_id}">Copy</button>
    <script>
    const btn = document.getElementById("btn_{button_id}");
    btn.onclick = () => {{
        navigator.clipboard.writeText('{safe_text}').then(()=> {{
            btn.innerText='Copied';
            setTimeout(()=> btn.innerText='Copy', 1200);
        }});
    }};
    </script>
    """
    components.html(component, height=40)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("<div style='text-align:center;margin-bottom:20px'><h2 style='color:#1f4ed8'>📚 UniAssist</h2></div>", unsafe_allow_html=True)
    nav = st.radio("📍 Choose a Section:", ["🏠 Home", "📚 Browse FAQ", "⭐ Bookmarks", "📝 Feedback", "🔐 Admin"], index=["🏠 Home","📚 Browse FAQ","⭐ Bookmarks","📝 Feedback","🔐 Admin"].index(st.session_state.get('nav_menu', "🏠 Home")) if st.session_state.get('nav_menu') in ["🏠 Home","📚 Browse FAQ","⭐ Bookmarks","📝 Feedback","🔐 Admin"] else 0, key="nav_menu")
    st.divider()
    # Quick stats intentionally removed but available as comment / future
    st.markdown("<div class='sidebar-section'><b>📖 Dataset Info</b><br>", unsafe_allow_html=True)
    st.write(f"• **Total Q&A:** {len(q)}")
    st.write(f"• **Categories:** {len(categories)}")
    st.write(f"• **Status:** ✅ Active")
    st.markdown("</div>", unsafe_allow_html=True)
    with st.expander("📋 Using UniAssist"):
        st.markdown("""
        **Tips for best results:**
        1️⃣ Be Specific — Use detailed questions  
        2️⃣ Use Keywords — Include relevant terms  
        3️⃣ Browse FAQ — Explore by category  
        4️⃣ Save Answers — Bookmark important Q&As  
        5️⃣ Rate Answers — Help us improve
        """)
    with st.expander("ℹ️ About"):
        st.markdown("""
        **UniAssist v2.0**  
        ✨ Semantic search  
        🔖 Save bookmarks  
        💬 Feedback system  
        🔐 Admin panel  
        © 2026 UniAssist
        """)

# expose nav to session_state so other buttons can switch to feedback
st.session_state['nav_menu'] = nav

# ---------------- Header ----------------
st.markdown("<div class='main-title'>🎓 UniAssist</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-Powered Academic Guidance Assistant</div>", unsafe_allow_html=True)
st.divider()

# ---------------- HOME implementation ----------------
if nav == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🤔 Ask Your Question")
        query = st.text_input("Enter your question:", placeholder="What is the minimum attendance requirement?", key="ex_query", label_visibility="collapsed")
        if st.button("🔍 Get Answer", use_container_width=True, type="primary"):
            if not query.strip():
                st.error("❌ Please enter a question first.")
            else:
                q_in = query.strip()[:500]
                with st.spinner("🔄 Searching knowledge base..."):
                    ans, conf, rel = search(q_in)
                if ans:
                    # record bounded history
                    st.session_state.search_history.append({'query': q_in, 'confidence': conf, 'timestamp': datetime.now().isoformat()})
                    if len(st.session_state.search_history) > 200:
                        st.session_state.search_history.pop(0)
                    st.markdown(f"<div class='answer-box'>{html.escape(ans)}</div>", unsafe_allow_html=True)
                    color = "confidence-high" if conf >= 0.7 else "confidence-medium" if conf >= 0.5 else "confidence-low"
                    text = "🟢 High" if conf >= 0.7 else "🟡 Medium" if conf >= 0.5 else "🔴 Low"
                    st.markdown(f"<p style='text-align:center;'><span class='{color}'>{text} Confidence: {conf:.0%}</span></p>", unsafe_allow_html=True)
                    # unique id for keys
                    unique = f"{abs(hash(q_in))}_{int(datetime.now().timestamp())%100000}"
                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        if st.button("⭐ Bookmark", key=f"bm_home_{unique}"):
                            if add_bookmark(q_in, ans):
                                st.success("✅ Saved to bookmarks!")
                            else:
                                st.info("✓ Already bookmarked")
                    with c2:
                        copy_button(ans, f"home_{unique}")
                    with c3:
                        if st.button("🗳️ Rate", key=f"rate_home_{unique}"):
                            # switch to feedback tab and prefill
                            st.session_state['nav_menu'] = "📝 Feedback"
                            st.session_state.fb_q = q_in
                            st.experimental_rerun()
                    # Related
                    if rel:
                        st.markdown("### 🔗 Related Questions")
                        for i, r in enumerate(rel, 1):
                            with st.expander(f"Q{i}: {r['q'][:70]}... ({int(r['s']*100)}%)"):
                                st.write(r['a'])
                                if st.button(f"⭐ Save Q{i}", key=f"save_rel_{unique}_{i}"):
                                    if add_bookmark(r['q'], r['a']):
                                        st.success("✅ Saved!")
                                    else:
                                        st.info("✓ Already bookmarked")
                else:
                    st.warning("⚠️ No reliable answer found. Try browsing FAQ or provide more details.")
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

# ---------------- FAQ ----------------
elif nav == "📚 Browse FAQ":
    st.subheader("📚 Frequently Asked Questions")
    col1, col2 = st.columns([2, 1])
    with col1:
        cat = st.selectbox("📂 Filter by Category:", ["All Categories"] + categories, key="faq_cat")
    with col2:
        per_page = st.selectbox("📄 Per Page:", [5, 10, 20, 50], index=1, key="faq_per")
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
        if st.button("◀ Previous", key="faqp", use_container_width=True):
            if st.session_state.faq_page > 0:
                st.session_state.faq_page -= 1
                st.experimental_rerun()
    with col_pg:
        st.markdown(f"<p style='text-align:center;font-weight:600'>Page {st.session_state.faq_page + 1} of {pages}</p>", unsafe_allow_html=True)
    with col_n:
        if st.button("Next ▶", key="faqn", use_container_width=True):
            if st.session_state.faq_page < pages - 1:
                st.session_state.faq_page += 1
                st.experimental_rerun()
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
                if st.button("⭐", key=f"faq_bm_{i}"):
                    if add_bookmark(q[i], a[i]):
                        st.success("✅ Bookmarked!")
                    else:
                        st.info("✓ Already bookmarked")
    else:
        st.info("📭 No questions in this category")

# ---------------- Bookmarks ----------------
elif nav == "⭐ Bookmarks":
    st.subheader("⭐ My Saved Questions")
    if st.session_state.bookmarks:
        cols_display = st.columns([1, 4])
        with cols_display[0]:
            st.write(f"**Total:** {len(st.session_state.bookmarks)}")
        for i, bm in enumerate(st.session_state.bookmarks):
            col1, col2 = st.columns([5, 1])
            with col1:
                with st.expander(f"📌 {bm.get('question','')[:75]}"):
                    st.write(bm.get('answer',''))
                    st.caption(f"Saved: {bm.get('timestamp','')[:10]}")
            with col2:
                if st.button("🗑️", key=f"rm_bm_{i}"):
                    remove_bookmark(bm.get('question'))
                    st.experimental_rerun()
    else:
        st.info("📌 No bookmarks yet. Save questions from Home or Browse FAQ!")

# ---------------- Feedback ----------------
elif nav == "📝 Feedback":
    st.subheader("📝 Provide Feedback")
    with st.form("feedback_form", clear_on_submit=False):
        fb_q = st.text_input("Question (optional):", value=st.session_state.get('fb_q', ''), key="feedback_q_input")
        fb_c = st.text_area("Your Feedback:", value=st.session_state.get('fb_c', ''), key="feedback_c_input", height=140)
        fb_r = st.slider("Rating", 1, 5, value=st.session_state.get('fb_r', 3), key="feedback_r_input")
        submitted = st.form_submit_button("✅ Submit Feedback")
        if submitted:
            if fb_c.strip():
                add_feedback(fb_q, fb_r, fb_c)
                st.success("✅ Thank you for your feedback!")
                st.session_state.fb_q = ""
                st.session_state.fb_c = ""
                st.session_state.fb_r = 3
                st.experimental_rerun()
            else:
                st.error("❌ Please add your feedback before submitting")
    col1, col2 = st.columns([3,1])
    with col2:
        if st.button("🔄 Clear", key="feedback_clear"):
            st.session_state.fb_q = ""
            st.session_state.fb_c = ""
            st.session_state.fb_r = 3
            st.experimental_rerun()
    if st.session_state.feedback_data:
        st.divider()
        st.markdown("### 📊 Feedback Summary")
        rates = [f.get('rating') for f in st.session_state.feedback_data if isinstance(f.get('rating'), int)]
        if rates:
            c1, c2, c3 = st.columns(3)
            c1.metric("Average Rating", f"{np.mean(rates):.1f}/5.0")
            c2.metric("Total Responses", len(rates))
            c3.metric("Latest", rates[-1] if rates else "N/A")

# ---------------- Admin ----------------
elif nav == "🔐 Admin":
    st.subheader("🔐 Admin Panel")
    if not st.session_state.admin_mode:
        st.warning("🔒 Password required to access admin panel")
        pwd = st.text_input("Enter Admin Password:", type="password", key="admin_pwd_input")
        if st.button("🔓 Login", key="admin_login"):
            if pwd == st.session_state.admin_password:
                st.session_state.admin_mode = True
                st.experimental_rerun()
            else:
                st.error("❌ Wrong password!")
    else:
        col1, col2 = st.columns([4,1])
        with col1:
            st.success("✅ Admin Mode Active")
        with col2:
            if st.button("🔒 Logout", key="admin_logout"):
                st.session_state.admin_mode = False
                st.experimental_rerun()
        st.divider()
        t1, t2, t3, t4 = st.tabs(["➕ Add Q&A", "📋 Manage", "💬 Feedback", "📤 Export"])
        with t1:
            st.markdown("### Add New Q&A Pair")
            nq = st.text_input("Question:", value=st.session_state.get('new_q',''), key="admin_q_input")
            na = st.text_area("Answer:", value=st.session_state.get('new_a',''), key="admin_a_input", height=120)
            cat_ch = st.radio("Add to:", ["Existing Category", "New Category"], key="cat_ch", horizontal=True)
            if cat_ch == "Existing Category":
                nc = st.selectbox("Select Category:", [""] + categories, key="Admin_cat")
            else:
                nc = st.text_input("New Category Name:", value=st.session_state.get('new_cat_name',''), key="admin_cat_input")
            if st.button("➕ Add Q&A", key="admin_add"):
                if nq.strip() and na.strip() and str(nc).strip():
                    st.session_state.custom_qa_pairs.append({
                        'question': nq.strip(), 'answer': na.strip(), 'category': str(nc).strip(), 'added_on': datetime.now().isoformat()
                    })
                    save_data()
                    # clear and rebuild embeddings cache safely
                    try:
                        st.cache_resource.clear()
                    except Exception:
                        pass
                    st.success("✅ Q&A Added Successfully! (Indexed)")
                    st.session_state.new_q = ""
                    st.session_state.new_a = ""
                    st.session_state.new_cat_name = ""
                    st.experimental_rerun()
                else:
                    st.error("❌ All fields are required")
        with t2:
            st.markdown("### Manage Custom Q&A")
            if st.session_state.custom_qa_pairs:
                st.write(f"**Total Custom Q&A:** {len(st.session_state.custom_qa_pairs)}")
                for i, qa in enumerate(st.session_state.custom_qa_pairs.copy()):
                    col1, col2 = st.columns([5,1])
                    with col1:
                        with st.expander(f"Q: {qa.get('question','')[:70]}..."):
                            st.write(f"**Answer:** {qa.get('answer','')}")
                            st.caption(f"Category: {qa.get('category','')} | Added: {qa.get('added_on','')[:10]}")
                    with col2:
                        if st.button("🗑️ Delete", key=f"del_qa_{i}"):
                            st.session_state.custom_qa_pairs = [x for x in st.session_state.custom_qa_pairs if x.get('question') != qa.get('question')]
                            save_data()
                            st.experimental_rerun()
            else:
                st.info("No custom Q&A pairs yet")
        with t3:
            st.markdown("### Feedback Analysis")
            if st.session_state.feedback_data:
                try:
                    df_fb = pd.DataFrame(st.session_state.feedback_data)
                    cols_show = [col for col in df_fb.columns if col in ['query', 'rating', 'comment', 'timestamp']]
                    st.dataframe(df_fb[cols_show] if cols_show else df_fb, use_container_width=True)
                except Exception:
                    st.write("Unable to render feedback table.")
                rates = [f.get('rating') for f in st.session_state.feedback_data if isinstance(f.get('rating'), int)]
                if rates:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg Rating", f"{np.mean(rates):.1f}/5")
                    col2.metric("Total", len(rates))
                    col3.metric("Highest", max(rates))
                if st.button("🗑️ Clear All Feedback", key="clear_fb"):
                    st.session_state.feedback_data = []
                    save_data()
                    st.success("✅ Feedback cleared")
                    st.experimental_rerun()
            else:
                st.info("No feedback yet")
        with t4:
            st.markdown("### Export Data")
            st.info(f"📊 Bookmarks: {len(st.session_state.bookmarks)} | 💬 Feedback: {len(st.session_state.feedback_data)} | ❓ Q&A: {len(st.session_state.custom_qa_pairs)}")
            fmt = st.selectbox("Export Format:", ["JSON", "CSV"], key="exp_fmt")
            data = {'bookmarks': st.session_state.bookmarks, 'feedback': st.session_state.feedback_data, 'custom_qa': st.session_state.custom_qa_pairs, 'exported': datetime.now().isoformat()}
            if fmt == "JSON":
                st.download_button("⬇️ Download JSON", json.dumps(data, indent=2), "uniassist_data.json", "application/json")
            else:
                if st.session_state.feedback_data:
                    csv = pd.DataFrame(st.session_state.feedback_data).to_csv(index=False)
                    st.download_button("⬇️ Download CSV", csv, "uniassist_feedback.csv", "text/csv")
                else:
                    st.info("No feedback data to export")

# ---------------- Footer ----------------
st.divider()
st.markdown("<div class='footer'>© 2026 UniAssist | AI-Powered Academic Guidance Assistant</div>", unsafe_allow_html=True)
