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

# Session state defaults
defaults = {'search_history': [], 'bookmarks': [], 'feedback_data': [], 'admin_mode': False,
            'admin_password': "admin123", 'custom_qa_pairs': [], 'rate_limit_count': 0,
            'faq_page': 0, 'fb_q': '', 'fb_c': '', 'fb_r': 3}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Minimal CSS
st.markdown("""<style>
body{background:#fff;color:#000}.main-title{font-size:40px;font-weight:700;color:#1f4ed8;text-align:center;margin-bottom:10px}
.sub-title{font-size:16px;color:#333;text-align:center;margin-bottom:20px}.answer-box{background:#e6f4ff;padding:20px;
border-radius:10px;border-left:6px solid #1f4ed8;color:#000;margin:15px 0}.confidence-high{color:#2ecc71;font-weight:bold}
.confidence-medium{color:#f39c12;font-weight:bold}.confidence-low{color:#e74c3c;font-weight:bold}.footer{font-size:12px;
color:#555;text-align:center;margin-top:50px}</style>""", unsafe_allow_html=True)

# Data persistence
def save_data():
    try:
        with open('uniassist_data.json', 'w') as f:
            json.dump({'bookmarks': st.session_state.bookmarks, 'feedback': st.session_state.feedback_data,
                       'custom_qa': st.session_state.custom_qa_pairs}, f)
    except Exception as e:
        st.error(f"❌ Save error: {e}")

def load_data_from_file():
    if os.path.exists('uniassist_data.json'):
        try:
            with open('uniassist_data.json', 'r') as f:
                data = json.load(f)
                st.session_state.bookmarks = data.get('bookmarks', [])
                st.session_state.feedback_data = data.get('feedback', [])
                st.session_state.custom_qa_pairs = data.get('custom_qa', [])
        except: pass

load_data_from_file()

# Load models
@st.cache_data
def load_qa():
    try:
        df = pd.read_csv("UniAssist_training_data.csv")
        return df["question"].tolist(), df["answer"].tolist(), df["category_name"].tolist()
    except:
        return [], [], []

@st.cache_resource
def load_model_embed():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except:
        return None

q, a, c = load_qa()
if st.session_state.custom_qa_pairs:
    for qa in st.session_state.custom_qa_pairs:
        q.append(qa.get('question', ''))
        a.append(qa.get('answer', ''))
        c.append(qa.get('category', 'Custom'))

model = load_model_embed()
embeddings = model.encode(q) if model and q else np.array([])
categories = sorted(set(cat for cat in c if cat))

THRESHOLD = 0.50

def search(query):
    if st.session_state.rate_limit_count > 100:
        st.error("❌ Rate limit exceeded")
        return None, 0, []
    if not model or embeddings.size == 0 or len(q) == 0:
        return None, 0, []
    st.session_state.rate_limit_count += 1
    try:
        query_vec = model.encode([query])
        scores = cosine_similarity(query_vec, embeddings)[0]
        top_idx = np.argmax(scores)
        if scores[top_idx] < THRESHOLD:
            return None, 0, []
        related = []
        for i in np.argsort(scores)[::-1][1:4]:
            if scores[i] >= (THRESHOLD - 0.05):
                related.append({'q': q[i], 'a': a[i], 's': scores[i]})
        return a[top_idx], scores[top_idx], related
    except:
        return None, 0, []

def add_bookmark(qn, ans):
    if not any(b['question'] == qn for b in st.session_state.bookmarks):
        st.session_state.bookmarks.append({'timestamp': datetime.now().isoformat(), 'question': qn, 'answer': ans})
        save_data()
        return True
    return False

def remove_bookmark(qn):
    st.session_state.bookmarks = [b for b in st.session_state.bookmarks if b['question'] != qn]
    save_data()
    return True

def add_feedback(qry, rate, comm=""):
    st.session_state.feedback_data.append({'timestamp': datetime.now().isoformat(), 'query': qry, 'rating': int(rate), 'comment': comm})
    save_data()

# Sidebar
with st.sidebar:
    st.title("📚 Navigation")
    nav = st.radio("Go to:", ["🏠 Home", "📚 Browse FAQ", "⭐ Bookmarks", "📝 Feedback", "🔐 Admin"], key="nav_menu")
    st.divider()
    with st.expander("ℹ️ About"):
        st.markdown("**UniAssist** - Academic Guidance\n\n✨ Semantic search • 🎓 1075+ Q&A • 🔖 Bookmarks • 💬 Feedback • 🔐 Admin")

st.markdown("<div class='main-title'>🎓 UniAssist</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Academic Guidance Assistant</div>", unsafe_allow_html=True)
st.divider()

# HOME
if nav == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🤔 Ask Your Question")
        query = st.text_input("", placeholder="What is the minimum attendance?", key="q", label_visibility="collapsed")
        col_s, col_e = st.columns([3, 1])
        with col_s:
            btn = st.button("🔍 Get Answer", use_container_width=True)
        with col_e:
            if st.button("?", use_container_width=True):
                st.session_state.q = "What is the minimum attendance requirement?"
                st.rerun()
        
        if btn and query.strip():
            query = query.strip()[:500]
            with st.spinner("🔄 Searching..."):
                ans, conf, rel = search(query)
            if ans:
                st.session_state.search_history.append({'query': query, 'confidence': conf, 'timestamp': datetime.now().isoformat()})
                st.markdown(f"<div class='answer-box'>{html.escape(ans)}</div>", unsafe_allow_html=True)
                color = "confidence-high" if conf >= 0.7 else "confidence-medium" if conf >= 0.5 else "confidence-low"
                text = "🟢 High" if conf >= 0.7 else "🟡 Medium" if conf >= 0.5 else "🔴 Low"
                st.markdown(f"<p style='text-align: center;'><span class='{color}'>{text}: {conf:.0%}</span></p>", unsafe_allow_html=True)
                
                cols = st.columns(5)
                with cols[0]:
                    if st.button("⭐ Bookmark", use_container_width=True, key="bm"):
                        if add_bookmark(query, ans):
                            st.success("✅ Saved!")
                        else:
                            st.info("✓ Already saved")
                with cols[1]:
                    if st.button("📋 Copy", use_container_width=True, key="cp"):
                        st.info("✓ Use browser copy")
                with cols[4]:
                    if st.button("👍 Rate", use_container_width=True, key="rt"):
                        st.info("→ Go to Feedback")
                
                if rel:
                    st.markdown("### 🔗 Related Questions")
                    for i, r in enumerate(rel, 1):
                        with st.expander(f"Q{i}: {r['q'][:55]}... ({r['s']:.0%})"):
                            st.write(r['a'])
                            if st.button(f"⭐ Save Q{i}", key=f"save_rel_{i}", use_container_width=True):
                                if add_bookmark(r['q'], r['a']):
                                    st.success("✅ Saved!")
                                else:
                                    st.info("✓ Already saved")
            else:
                st.error("❌ No reliable answer found. Try browsing FAQ or provide more details.")
        elif btn:
            st.warning("⚠️ Please enter a question.")
    with col2:
        st.markdown("### 📊 Stats")
        st.metric("Searches", len(st.session_state.search_history))
        st.metric("Bookmarks", len(st.session_state.bookmarks))
        st.metric("Ratings", len(st.session_state.feedback_data))
        if st.session_state.search_history:
            avg_conf = np.mean([h.get('confidence', 0) for h in st.session_state.search_history])
            st.metric("Avg Confidence", f"{avg_conf:.0%}")

# BROWSE FAQ
elif nav == "📚 Browse FAQ":
    st.subheader("📚 Browse FAQ")
    col1, col2 = st.columns([2, 1])
    with col1:
        cat = st.selectbox("Category:", ["All"] + categories, key="faq_cat")
    with col2:
        per_page = st.selectbox("Per page:", [5, 10, 20, 50], key="faq_per")
    
    if not hasattr(st.session_state, '_faq_cat'):
        st.session_state._faq_cat = cat
        st.session_state._faq_per = per_page
    if cat != st.session_state._faq_cat or per_page != st.session_state._faq_per:
        st.session_state.faq_page = 0
        st.session_state._faq_cat, st.session_state._faq_per = cat, per_page
    
    idx = list(range(len(q))) if cat == "All" else [i for i, x in enumerate(c) if x == cat]
    pages = max(1, (len(idx) + per_page - 1) // per_page)
    if st.session_state.faq_page >= pages:
        st.session_state.faq_page = 0
    
    col_p, col_pg, col_n = st.columns([1, 2, 1])
    with col_p:
        if st.button("◀ Prev", use_container_width=True, key="faqp"):
            if st.session_state.faq_page > 0:
                st.session_state.faq_page -= 1
                st.rerun()
    with col_pg:
        st.markdown(f"<p style='text-align:center'>Page {st.session_state.faq_page + 1}/{pages}</p>", unsafe_allow_html=True)
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
                with st.expander(f"{q[i][:70]}..."):
                    st.write(a[i])
                    st.caption(f"Category: {c[i]}")
            with col_bm:
                if st.button("⭐", key=f"faq_bm_{i}"):
                    if add_bookmark(q[i], a[i]):
                        st.success("✓")
                    else:
                        st.info("✓")
    else:
        st.info("No questions in this category")

# BOOKMARKS
elif nav == "⭐ Bookmarks":
    st.subheader("⭐ My Bookmarks")
    if st.session_state.bookmarks:
        for i, bm in enumerate(st.session_state.bookmarks):
            col1, col2 = st.columns([5, 1])
            with col1:
                with st.expander(f"{bm['question'][:70]}..."):
                    st.write(bm['answer'])
                    st.caption(f"Saved: {bm['timestamp'][:10]}")
            with col2:
                if st.button("🗑️", key=f"rm_bm_{i}"):
                    if remove_bookmark(bm['question']):
                        st.rerun()
    else:
        st.info("📌 No bookmarks yet")

# FEEDBACK
elif nav == "📝 Feedback":
    st.subheader("📝 Feedback")
    col1, col2 = st.columns([2, 1])
    with col1:
        qry = st.text_input("Question:", key="fb_q", label_visibility="collapsed", placeholder="(Optional)")
        cmt = st.text_area("Comment:", key="fb_c", label_visibility="collapsed", height=100)
    with col2:
        rate = st.radio("Rate:", [1, 2, 3, 4, 5], key="fb_r")
    
    if st.button("✅ Submit", type="primary", use_container_width=True):
        if cmt.strip():
            add_feedback(qry, rate, cmt)
            st.success("✅ Thank you for your feedback!")
            st.session_state.fb_q = ""
            st.session_state.fb_c = ""
            st.session_state.fb_r = 3
            st.rerun()
        else:
            st.error("❌ Please add your feedback")
    
    if st.session_state.feedback_data:
        st.divider()
        st.markdown("### 📊 Summary")
        rates = [f['rating'] for f in st.session_state.feedback_data if isinstance(f.get('rating'), int)]
        if rates:
            st.metric("Average Rating", f"{np.mean(rates):.1f}/5", f"{len(rates)} responses")

# ADMIN
elif nav == "🔐 Admin":
    st.subheader("🔐 Admin Panel")
    if not st.session_state.admin_mode:
        pwd = st.text_input("Password:", type="password", key="admin_pwd")
        if st.button("Login", key="admin_btn"):
            if pwd == st.session_state.admin_password:
                st.session_state.admin_mode = True
                st.rerun()
            else:
                st.error("❌ Wrong password!")
    else:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.success("✅ Admin Mode Active")
        with col2:
            if st.button("Logout"):
                st.session_state.admin_mode = False
                st.rerun()
        
        st.divider()
        t1, t2, t3, t4 = st.tabs(["Add Q&A", "Manage", "Feedback", "Export"])
        
        with t1:
            st.markdown("### Add Q&A")
            nq = st.text_input("Q:", key="new_q", label_visibility="collapsed")
            na = st.text_area("A:", key="new_a", label_visibility="collapsed")
            cat_ch = st.radio("Add to:", ["Existing", "New"], key="cat_ch", horizontal=True)
            nc = st.selectbox("Cat:", categories, key="new_cat") if cat_ch == "Existing" else st.text_input("New cat:", key="new_cat_name")
            
            if st.button("➕ Add Q&A", type="primary", use_container_width=True):
                if nq.strip() and na.strip() and nc.strip():
                    st.session_state.custom_qa_pairs.append({'question': nq.strip(), 'answer': na.strip(), 'category': nc.strip(), 'added_on': datetime.now().isoformat()})
                    save_data()
                    st.success("✅ Q&A added! (Reload to index)")
                    st.rerun()
                else:
                    st.error("❌ All fields required")
        
        with t2:
            st.markdown("### Custom Q&A")
            if st.session_state.custom_qa_pairs:
                for i in range(len(st.session_state.custom_qa_pairs)):
                    qa = st.session_state.custom_qa_pairs[i]
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        with st.expander(f"{qa['question'][:60]}..."):
                            st.write(f"**A:** {qa['answer']}")
                            st.caption(f"Cat: {qa['category']}")
                    with col2:
                        if st.button("🗑️", key=f"del_{i}"):
                            st.session_state.custom_qa_pairs.pop(i)
                            save_data()
                            st.rerun()
            else:
                st.info("No custom Q&A yet")
        
        with t3:
            st.markdown("### Feedback")
            if st.session_state.feedback_data:
                df_fb = pd.DataFrame(st.session_state.feedback_data)
                cols_show = [col for col in df_fb.columns if col in ['query', 'rating', 'comment', 'timestamp']]
                if cols_show:
                    st.dataframe(df_fb[cols_show], use_container_width=True)
                rates = [f['rating'] for f in st.session_state.feedback_data if isinstance(f.get('rating'), int)]
                if rates:
                    st.metric("Average Rating", f"{np.mean(rates):.1f}/5", f"{len(rates)} responses")
                if st.button("🗑️ Clear All", key="clear_fb"):
                    st.session_state.feedback_data = []
                    save_data()
                    st.success("✅ Feedback cleared")
                    st.rerun()
            else:
                st.info("No feedback yet")
        
        with t4:
            st.markdown("### Export Data")
            st.write(f"📊 Bookmarks: {len(st.session_state.bookmarks)} | 💬 Feedback: {len(st.session_state.feedback_data)} | ❓ Q&A: {len(st.session_state.custom_qa_pairs)}")
            
            fmt = st.selectbox("Format:", ["JSON", "CSV"], key="exp_fmt")
            data = {'bookmarks': st.session_state.bookmarks, 'feedback': st.session_state.feedback_data,
                    'custom_qa': st.session_state.custom_qa_pairs, 'exported': datetime.now().isoformat()}
            
            if fmt == "JSON":
                st.download_button("⬇️ Download JSON", json.dumps(data, indent=2), "uniassist_export.json", "application/json", use_container_width=True)
            else:
                if st.session_state.feedback_data:
                    csv = pd.DataFrame(st.session_state.feedback_data).to_csv(index=False)
                    st.download_button("⬇️ Download CSV", csv, "uniassist_feedback.csv", "text/csv", use_container_width=True)
                else:
                    st.info("No feedback data to export")

st.divider()
st.markdown("<div class='footer'>© 2026 UniAssist | Academic Guidance</div>", unsafe_allow_html=True)
