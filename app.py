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
            'faq_page': 0, 'feedback_query_input': '', 'feedback_comment_input': '', 'feedback_rating_input': 3}
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
    with open('uniassist_data.json', 'w') as f:
        json.dump({'bookmarks': st.session_state.bookmarks, 'feedback': st.session_state.feedback_data,
                   'custom_qa': st.session_state.custom_qa_pairs}, f)

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
    if not model or embeddings.size == 0 or st.session_state.rate_limit_count > 100:
        return None, 0, []
    st.session_state.rate_limit_count += 1
    scores = cosine_similarity(model.encode([query]), embeddings)[0]
    top_idx = np.argmax(scores)
    if scores[top_idx] < THRESHOLD:
        return None, 0, []
    related = []
    for i in np.argsort(scores)[::-1][1:4]:
        if scores[i] >= (THRESHOLD - 0.05):
            related.append({'q': q[i], 'a': a[i], 's': scores[i]})
    return a[top_idx], scores[top_idx], related

def add_bookmark(qn, ans):
    if not any(b['question'] == qn for b in st.session_state.bookmarks):
        st.session_state.bookmarks.append({'timestamp': datetime.now().isoformat(), 'question': qn, 'answer': ans})
        save_data()
        return True
    return False

def remove_bookmark(qn):
    st.session_state.bookmarks = [b for b in st.session_state.bookmarks if b['question'] != qn]
    save_data()

def add_feedback(qry, rate, comm=""):
    st.session_state.feedback_data.append({'timestamp': datetime.now().isoformat(), 'query': qry, 'rating': int(rate), 'comment': comm})
    save_data()

# Sidebar
with st.sidebar:
    st.title("📚 Navigation")
    nav = st.radio("Go to:", ["🏠 Home", "📚 Browse FAQ", "🔍 Advanced Search", "⭐ Bookmarks", "📝 Feedback", "🔐 Admin"], key="nav_menu")
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
            with st.spinner("Searching..."):
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
                        add_bookmark(query, ans) and st.success("Added!") or st.info("Already saved")
                with cols[1]:
                    st.button("📋 Copy", use_container_width=True, key="cp")
                with cols[4]:
                    st.button("👍 Rate", use_container_width=True, key="rt")
                
                if rel:
                    st.markdown("### 🔗 Related")
                    for i, r in enumerate(rel, 1):
                        with st.expander(f"Q{i}: {r['q'][:50]}... ({r['s']:.0%})"):
                            st.write(r['a'])
            else:
                st.error("No reliable answer found. Try advanced search or browse FAQ.")
        elif btn:
            st.warning("Please enter a question.")
    with col2:
        st.markdown("### 📊 Stats")
        st.metric("Bookmarks", len(st.session_state.bookmarks))
        st.metric("Feedback", len(st.session_state.feedback_data))
        st.metric("Searches", len(st.session_state.search_history))

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
            st.session_state.faq_page = max(0, st.session_state.faq_page - 1)
            st.rerun()
    with col_pg:
        st.markdown(f"<p style='text-align:center'>Page {st.session_state.faq_page + 1}/{pages}</p>", unsafe_allow_html=True)
    with col_n:
        if st.button("Next ▶", use_container_width=True, key="faqn"):
            st.session_state.faq_page = min(pages - 1, st.session_state.faq_page + 1)
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
                    add_bookmark(q[i], a[i]) and st.success("✓") or st.info("✓")
    else:
        st.info("No questions")

# ADVANCED SEARCH
elif nav == "🔍 Advanced Search":
    st.subheader("🔍 Advanced Search")
    search_type = st.radio("Type:", ["Keywords", "Category", "Similarity"], horizontal=True, key="srch_type")
    
    if search_type == "Keywords":
        kw = st.text_input("Keywords:", key="kw_srch")
        min_kw = st.slider("Min matches:", 1, 5, 1, key="min_kw")
        if kw:
            kws = kw.lower().split()
            res = [(i, sum(1 for k in kws if k in q[i].lower())) for i in range(len(q))]
            res = sorted([x for x in res if x[1] >= min_kw], key=lambda x: x[1], reverse=True)[:20]
            if res:
                st.success(f"Found {len(res)} results")
                for i, cnt in res:
                    with st.expander(f"{q[i][:70]}... ({cnt} matches)"):
                        st.write(a[i])
            else:
                st.warning("No matches")
    
    elif search_type == "Category":
        cats = st.multiselect("Select:", categories, key="cat_srch")
        if cats:
            for cat in cats:
                st.markdown(f"### {cat}")
                items = [(i, q[i]) for i in range(len(q)) if c[i] == cat][:10]
                for i, qn in items:
                    with st.expander(f"{qn[:70]}..."):
                        st.write(a[i])
    
    elif search_type == "Similarity":
        ref = st.text_area("Reference question:", key="sim_ref")
        num = st.slider("Results:", 1, 20, 5, key="sim_num")
        if ref and model and embeddings.size:
            scores = cosine_similarity(model.encode([ref]), embeddings)[0]
            top_i = np.argsort(scores)[::-1][:num]
            for rank, i in enumerate(top_i, 1):
                with st.expander(f"#{rank} {q[i][:70]}... ({scores[i]:.0%})"):
                    st.write(a[i])

# BOOKMARKS
elif nav == "⭐ Bookmarks":
    st.subheader("⭐ Bookmarks")
    if st.session_state.bookmarks:
        for i, bm in enumerate(st.session_state.bookmarks):
            col1, col2 = st.columns([5, 1])
            with col1:
                with st.expander(f"{bm['question'][:70]}..."):
                    st.write(bm['answer'])
                    st.caption(f"Saved: {bm['timestamp'][:10]}")
            with col2:
                if st.button("🗑️", key=f"rm_bm_{i}"):
                    remove_bookmark(bm['question'])
                    st.rerun()
    else:
        st.info("No bookmarks")

# FEEDBACK
elif nav == "📝 Feedback":
    st.subheader("📝 Feedback")
    col1, col2 = st.columns([2, 1])
    with col1:
        qry = st.text_input("Question:", key="fb_q", label_visibility="collapsed")
        cmt = st.text_area("Comment:", key="fb_c", label_visibility="collapsed", height=100)
    with col2:
        rate = st.radio("Rate:", [1, 2, 3, 4, 5], key="fb_r")
    
    if st.button("Submit", type="primary", use_container_width=True):
        if cmt.strip():
            add_feedback(qry, rate, cmt)
            st.success("✅ Thank you!")
            st.session_state.fb_q = ""
            st.session_state.fb_c = ""
            st.session_state.fb_r = 3
            st.rerun()
        else:
            st.error("Add comment")
    
    if st.session_state.feedback_data:
        st.divider()
        st.markdown("### 📊 Summary")
        rates = [f['rating'] for f in st.session_state.feedback_data if isinstance(f.get('rating'), int)]
        if rates:
            st.metric("Avg Rating", f"{np.mean(rates):.1f}/5", f"{len(rates)} responses")

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
                st.error("Wrong!")
    else:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.success("✅ Admin Mode")
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
            
            if st.button("Add", type="primary", use_container_width=True):
                if nq.strip() and na.strip() and nc.strip():
                    st.session_state.custom_qa_pairs.append({'question': nq, 'answer': na, 'category': nc, 'added_on': datetime.now().isoformat()})
                    save_data()
                    st.success("✅ Added! Reload to index")
                    st.rerun()
                else:
                    st.error("Fill all")
        
        with t2:
            st.markdown("### Custom Q&A")
            if st.session_state.custom_qa_pairs:
                for i, qa in enumerate(st.session_state.custom_qa_pairs):
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
                st.info("None")
        
        with t3:
            st.markdown("### Feedback")
            if st.session_state.feedback_data:
                st.dataframe(pd.DataFrame(st.session_state.feedback_data), use_container_width=True)
                if st.button("Clear", key="clear_fb"):
                    st.session_state.feedback_data = []
                    save_data()
                    st.rerun()
            else:
                st.info("None")
        
        with t4:
            st.markdown("### Export")
            fmt = st.selectbox("Format:", ["JSON", "CSV"], key="exp_fmt")
            data = {'bookmarks': st.session_state.bookmarks, 'feedback': st.session_state.feedback_data,
                    'custom_qa': st.session_state.custom_qa_pairs, 'exported': datetime.now().isoformat()}
            
            if fmt == "JSON":
                st.download_button("Download JSON", json.dumps(data, indent=2), "export.json", "application/json")
            else:
                if st.session_state.feedback_data:
                    csv = pd.DataFrame(st.session_state.feedback_data).to_csv(index=False)
                    st.download_button("Download CSV", csv, "feedback.csv", "text/csv")
                else:
                    st.info("No data")

st.divider()
st.markdown("<div class='footer'>© 2026 UniAssist | Academic Guidance</div>", unsafe_allow_html=True)
