
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
from io import BytesIO

# Try to import gtts for text-to-speech
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except:
    GTTS_AVAILABLE = False

# Try to import reportlab for PDF export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="UniAssist - Enhanced",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ SESSION STATE INITIALIZATION ============
session_defaults = {
    'theme': 'light',
    'language': 'English',
    'search_history': [],
    'bookmarks': [],
    'feedback_data': [],
    'admin_mode': False,
    'admin_password': "admin123",
    'custom_qa_pairs': [],
    'rate_limit_count': 0,
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ============ LANGUAGE CONFIGURATION ============
LANGUAGES = {
    'English': 'en',
    'Hindi': 'hi',
    'Spanish': 'es',
}

TRANSLATIONS = {
    'en': {
        'title': '🎓 UniAssist',
        'subtitle': 'Academic & Internship Guidance Assistant - Enhanced Edition',
        'ask_question': 'Ask your question',
        'get_answer': 'Get Answer',
        'answer': 'Answer',
        'confidence': 'Confidence Score',
        'related': 'Related Questions',
        'search_history': 'Search History',
        'bookmarks': 'Bookmarks',
        'feedback': 'Provide Feedback',
        'home': 'Home',
        'browse': 'Browse FAQ',
        'search': 'Advanced Search',
        'analytics': 'Analytics Dashboard',
        'admin': 'Admin Panel',
    },
    'hi': {
        'title': '🎓 यूनिएसिस्ट',
        'subtitle': 'शैक्षणिक और इंटर्नशिप मार्गदर्शन - उन्नत संस्करण',
        'ask_question': 'अपना सवाल पूछें',
        'get_answer': 'जवाब प्राप्त करें',
        'answer': 'जवाब',
        'confidence': 'आत्मविश्वास स्कोर',
        'related': 'संबंधित सवाल',
        'search_history': 'खोज इतिहास',
        'bookmarks': 'बुकमार्क',
        'feedback': 'प्रतिक्रिया दें',
        'home': 'होम',
        'browse': 'FAQ ब्राउज़ करें',
        'search': 'उन्नत खोज',
        'analytics': 'विश्लेषण डैशबोर्ड',
        'admin': 'व्यवस्थापक पैनल',
    }
}

# ============ CUSTOM CSS ============
def get_theme_css(theme):
    if theme == 'dark':
        return """
<style>
body { background-color: #1a1a1a; }
.main-title {
    font-size: 40px;
    font-weight: 700;
    color: #4a9eff;
    text-align: center;
    margin-bottom: 10px;
}
.sub-title {
    font-size: 16px;
    color: #aaa;
    text-align: center;
    margin-bottom: 20px;
}
.answer-box {
    background-color: #2d2d2d;
    padding: 20px;
    border-radius: 10px;
    border-left: 6px solid #4a9eff;
    color: #fff;
    margin: 15px 0;
}
.stats-box {
    background-color: #2d2d2d;
    padding: 20px;
    border-radius: 10px;
    color: #fff;
    border: 1px solid #444;
    margin: 10px 0;
}
.confidence-high {
    color: #2ecc71;
    font-weight: bold;
}
.confidence-medium {
    color: #f39c12;
    font-weight: bold;
}
.confidence-low {
    color: #e74c3c;
    font-weight: bold;
}
.footer {
    font-size: 12px;
    color: #666;
    text-align: center;
    margin-top: 50px;
}
.tag {
    display: inline-block;
    background-color: #4a9eff;
    color: white;
    padding: 5px 10px;
    border-radius: 15px;
    margin: 5px 5px 5px 0;
    font-size: 12px;
}
</style>
"""
    else:
        return """
<style>
.main-title {
    font-size: 40px;
    font-weight: 700;
    color: #1f4ed8;
    text-align: center;
    margin-bottom: 10px;
}
.sub-title {
    font-size: 16px;
    color: #555;
    text-align: center;
    margin-bottom: 20px;
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
.tag {
    display: inline-block;
    background-color: #1f4ed8;
    color: white;
    padding: 5px 10px;
    border-radius: 15px;
    margin: 5px 5px 5px 0;
    font-size: 12px;
}
</style>
"""

st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# ============ PERSISTENT DATA MANAGEMENT ============
def load_persistent_data():
    """Load bookmarks, feedback, and custom Q&A from JSON file"""
    if os.path.exists('uniassist_data.json'):
        try:
            with open('uniassist_data.json', 'r') as f:
                data = json.load(f)
                st.session_state.feedback_data = data.get('feedback', [])
                st.session_state.bookmarks = data.get('bookmarks', [])
                st.session_state.custom_qa_pairs = data.get('custom_qa', [])
        except:
            pass

def save_persistent_data():
    """Save bookmarks, feedback, and custom Q&A to JSON file"""
    data = {
        'feedback': st.session_state.feedback_data,
        'bookmarks': st.session_state.bookmarks,
        'custom_qa': st.session_state.custom_qa_pairs,
        'last_updated': datetime.now().isoformat()
    }
    try:
        with open('uniassist_data.json', 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving data: {e}")

load_persistent_data()

# ============ DATA LOADING ============
@st.cache_data
def load_data():
    """Load Q&A data from CSV"""
    df = pd.read_csv("UniAssist_training_data.csv")
    return df

@st.cache_resource
def load_model():
    """Load embedding model"""
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load base data
qa_frame = load_data()
questions = qa_frame["question"].astype(str).tolist()
answers = qa_frame["answer"].astype(str).tolist()
categories = qa_frame["category_name"].astype(str).tolist()

# Add custom Q&A pairs
if st.session_state.custom_qa_pairs:
    for qa in st.session_state.custom_qa_pairs:
        questions.append(qa['question'])
        answers.append(qa['answer'])
        categories.append(qa.get('category', 'Custom'))

model = load_model()
question_embeddings = model.encode(questions)

# Get unique categories
unique_categories = sorted(set(categories))

# ============ CONSTANTS & SETTINGS ============
SIMILARITY_THRESHOLD = 0.50
SAFE_FALLBACK_MESSAGE = (
    "I'm sorry, I don't have reliable information on this topic. "
    "UniAssist currently handles academic and internship-related queries only. "
    "Try using advanced search or browsing the FAQ."
)
RATE_LIMIT = 100

# ============ UTILITY FUNCTIONS ============

def sanitize_input(text):
    """Sanitize user input"""
    return str(text).strip()[:500]

def check_rate_limit():
    """Check rate limiting"""
    st.session_state.rate_limit_count += 1
    if st.session_state.rate_limit_count > RATE_LIMIT:
        st.error("Rate limit exceeded. Please try again later.")
        return False
    return True

def get_confidence_color(score):
    """Get confidence indicator"""
    if score >= 0.70:
        return "confidence-high", "🟢 High"
    elif score >= 0.50:
        return "confidence-medium", "🟡 Medium"
    else:
        return "confidence-low", "🔴 Low"

def get_answer_with_confidence(user_query, top_k=5):
    """Get answer with confidence score and related questions"""
    if not check_rate_limit():
        return None, None, []
    
    query_vec = model.encode([user_query])
    scores = cosine_similarity(query_vec, question_embeddings)[0]
    
    # Get top-k results
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[top_indices]
    
    best_index = top_indices[0]
    best_score = top_scores[0]
    
    if best_score < SIMILARITY_THRESHOLD:
        return None, None, []
    
    # Get related questions
    related = []
    for idx in top_indices[1:min(4, len(top_indices))]:
        if top_scores[top_indices.tolist().index(idx)] >= (SIMILARITY_THRESHOLD - 0.05):
            related.append({
                'question': questions[idx],
                'answer': answers[idx],
                'score': float(top_scores[top_indices.tolist().index(idx)])
            })
    
    return answers[best_index], float(best_score), related

def add_to_history(query, answer, confidence):
    """Add query to search history"""
    st.session_state.search_history.append({
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'answer': answer[:100] + "..." if len(answer) > 100 else answer,
        'confidence': confidence
    })

def add_bookmark(question, answer):
    """Add bookmark"""
    # Prevent duplicates
    if not any(b['question'] == question for b in st.session_state.bookmarks):
        st.session_state.bookmarks.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer
        })
        save_persistent_data()
        return True
    return False

def remove_bookmark(question):
    """Remove bookmark"""
    st.session_state.bookmarks = [b for b in st.session_state.bookmarks if b['question'] != question]
    save_persistent_data()

def add_feedback(query, rating, comment=""):
    """Store feedback"""
    st.session_state.feedback_data.append({
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'rating': rating,
        'comment': comment
    })
    save_persistent_data()

def get_related_by_category(category, limit=5):
    """Get related questions from same category"""
    related = []
    for i, q in enumerate(questions):
        if categories[i] == category and len(related) < limit:
            related.append((q, answers[i], categories[i]))
    return related

def get_analytics():
    """Generate analytics"""
    if not st.session_state.search_history:
        return None
    
    queries = [h['query'] for h in st.session_state.search_history]
    total_searches = len(queries)
    avg_confidence = np.mean([h['confidence'] for h in st.session_state.search_history])
    
    # Most searched words
    all_words = ' '.join(queries).lower().split()
    word_freq = Counter(all_words)
    
    # Filter common words
    common_words = {'what', 'is', 'the', 'a', 'and', 'or', 'for', 'to', 'in', 'of', 'how', 'can', 'i', 'about'}
    word_freq = {w: c for w, c in word_freq.items() if w not in common_words and len(w) > 2}
    
    # Feedback stats
    feedback = st.session_state.feedback_data
    avg_rating = np.mean([f['rating'] for f in feedback]) if feedback else 0
    
    return {
        'total_searches': total_searches,
        'avg_confidence': avg_confidence,
        'top_words': Counter(word_freq).most_common(5),
        'feedback_count': len(feedback),
        'avg_rating': avg_rating
    }

def export_to_pdf(question, answer):
    """Export answer to PDF"""
    if not PDF_AVAILABLE:
        return None
    
    try:
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        
        p.setFont("Helvetica-Bold", 16)
        p.drawString(50, 750, "UniAssist - Answer Export")
        
        p.setFont("Helvetica", 12)
        p.drawString(50, 720, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, 690, "Question:")
        p.setFont("Helvetica", 11)
        y = 670
        for line in question.split('\n'):
            if y < 100:
                p.showPage()
                y = 750
            p.drawString(70, y, line)
            y -= 20
        
        y -= 10
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, y, "Answer:")
        p.setFont("Helvetica", 11)
        y -= 20
        for line in answer.split('\n'):
            if y < 100:
                p.showPage()
                y = 750
            p.drawString(70, y, line)
            y -= 20
        
        p.setFont("Helvetica-Oblique", 9)
        p.drawString(50, 30, "UniAssist © 2026 | Academic Guidance Assistant")
        
        p.save()
        buffer.seek(0)
        return buffer
    except:
        return None

def text_to_speech(text, language='en'):
    """Convert text to speech"""
    if not GTTS_AVAILABLE:
        return None
    
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        audio = BytesIO()
        tts.write_to_fp(audio)
        audio.seek(0)
        return audio
    except:
        return None

# ============ SIDEBAR & NAVIGATION ============
with st.sidebar:
    st.title("⚙️ Settings & Navigation")
    
    # Theme & Language controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌙" if st.session_state.theme == 'light' else "☀️", help="Toggle theme", use_container_width=True):
            st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
            st.rerun()
    
    with col2:
        lang = st.selectbox("🌐 Language", options=list(LANGUAGES.keys()))
        if lang != st.session_state.language:
            st.session_state.language = lang
    
    st.divider()
    
    # Navigation
    nav_choice = st.radio(
        "Navigate to:",
        options=[
            "🏠 Home",
            "📚 Browse FAQ",
            "🔍 Advanced Search",
            "📊 Analytics",
            "⭐ Bookmarks",
            "📝 Feedback",
            "⚡ Quick Tips",
            "🔐 Admin Panel"
        ],
        key="nav_menu"
    )
    
    st.divider()
    
    # Info Section
    with st.expander("ℹ️ About UniAssist", expanded=False):
        st.markdown("""
        **UniAssist Enhanced** is an AI-powered guidance assistant with:
        
        ✨ Semantic Q&A matching
        📱 Multi-language support
        🌙 Dark/Light mode
        🔖 Bookmarks & History
        📊 Analytics dashboard
        💬 User feedback system
        📄 PDF export
        🔊 Text-to-speech
        """)
    
    with st.expander("📋 Features", expanded=False):
        st.markdown("""
        • **Smart Search**: AI-powered semantic matching
        • **FAQ Browser**: Browse by category
        • **History**: Track your searches
        • **Bookmarks**: Save important answers
        • **Analytics**: View search trends
        • **Feedback**: Help us improve
        • **Admin Panel**: Manage Q&A
        • **Multi-language**: English, Hindi, Spanish
        """)
    
    with st.expander("⚠️ Disclaimer", expanded=False):
        st.info("""
        This tool provides informational guidance only.
        For official decisions, always refer to university notifications.
        """)
    
    with st.expander("👤 About Author", expanded=False):
        st.markdown("**Mehul Kumar** - Developer\n\n*2026 Academic Project*")

# ============ MAIN HEADER ============
# Resolve language code robustly:
# - If the session value is a language name (e.g., 'English'), map to code via LANGUAGES
# - If the session value is already a language code (e.g., 'en'), use it
# - Fall back to English ('en') when mapping/translation is missing
raw_lang = st.session_state.language
if raw_lang in LANGUAGES:
    lang_code = LANGUAGES[raw_lang]
elif raw_lang in TRANSLATIONS:
    lang_code = raw_lang
else:
    lang_code = 'en'

trans = TRANSLATIONS.get(lang_code, TRANSLATIONS['en'])
st.markdown(f"<div class='main-title'>{trans.get('title','UniAssist')}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-title'>{trans.get('subtitle','Academic & Internship Guidance Assistant')}</div>", unsafe_allow_html=True)
st.divider()

# ============ PAGE ROUTING ============

# -------- HOME PAGE --------
if nav_choice == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🤔 Ask Your Question")
        
        user_query = st.text_input(
            "Enter your query",
            placeholder="e.g., What is the minimum attendance requirement?"
        )
        
        col_search, col_example = st.columns([3, 1])
        with col_search:
            ask_button = st.button("🔍 Get Answer", use_container_width=True)
        with col_example:
            if st.button("? Example", use_container_width=True):
                user_query = "What is the minimum attendance requirement?"
        
        if ask_button and user_query.strip():
            user_query = sanitize_input(user_query)
            
            with st.spinner("🔍 Searching for answer..."):
                answer, confidence, related = get_answer_with_confidence(user_query)
            
            if answer:
                add_to_history(user_query, answer, confidence)
                
                # Display answer
                st.markdown("### 📘 Answer")
                st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
                
                # Confidence score
                color_class, confidence_text = get_confidence_color(confidence)
                st.markdown(
                    f"<p style='text-align: center;'><span class='{color_class}'>{confidence_text}: {confidence:.1%}</span></p>",
                    unsafe_allow_html=True
                )
                
                # Action buttons
                action_col1, action_col2, action_col3, action_col4, action_col5 = st.columns(5)
                
                with action_col1:
                    if st.button("⭐ Bookmark", use_container_width=True):
                        if add_bookmark(user_query, answer):
                            st.success("Added to bookmarks!")
                        else:
                            st.info("Already bookmarked")
                
                with action_col2:
                    if st.button("📋 Copy", use_container_width=True):
                        st.write("Use Ctrl+C to copy the answer text")
                
                with action_col3:
                    if GTTS_AVAILABLE and st.button("🔊 Speak", use_container_width=True):
                        audio = text_to_speech(answer, LANGUAGES[st.session_state.language])
                        if audio:
                            st.audio(audio, format="audio/mp3")
                
                with action_col4:
                    if PDF_AVAILABLE and st.button("📄 PDF", use_container_width=True):
                        pdf = export_to_pdf(user_query, answer)
                        if pdf:
                            st.download_button(
                                label="Download PDF",
                                data=pdf,
                                file_name="uniassist_answer.pdf",
                                mime="application/pdf"
                            )
                
                with action_col5:
                    if st.button("👍 Rate", use_container_width=True):
                        st.info("Rating feature coming in feedback section")
                
                # Related questions
                if related:
                    st.markdown("### 🔗 Related Questions")
                    for i, rel in enumerate(related, 1):
                        with st.expander(f"Q{i}: {rel['question'][:60]}... ({rel['score']:.0%})"):
                            st.write(rel['answer'])
            else:
                st.error(SAFE_FALLBACK_MESSAGE)
        
        elif ask_button:
            st.warning("Please enter a question.")
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        stats = get_analytics()
        if stats:
            st.metric("Total Searches", stats['total_searches'])
            st.metric("Avg Confidence", f"{stats['avg_confidence']:.0%}")
            st.metric("Bookmarks", len(st.session_state.bookmarks))
            st.metric("Feedback Entries", stats['feedback_count'])
        else:
            st.info("No data yet. Start asking questions!")

# -------- BROWSE FAQ --------
elif nav_choice == "📚 Browse FAQ":
    st.subheader("📚 Browse Frequently Asked Questions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_category = st.selectbox(
            "Filter by category:",
            options=["All Categories"] + unique_categories
        )
    
    with col2:
        items_per_page = st.selectbox("Items per page:", [5, 10, 20, 50])
    
    # Filter Q&A
    if selected_category == "All Categories":
        filtered_indices = list(range(len(questions)))
    else:
        filtered_indices = [i for i, cat in enumerate(categories) if cat == selected_category]
    
    # Pagination
    total_items = len(filtered_indices)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    if 'faq_page' not in st.session_state:
        st.session_state.faq_page = 0
    
    col_prev, col_page, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("◀ Previous", use_container_width=True) and st.session_state.faq_page > 0:
            st.session_state.faq_page -= 1
    with col_page:
        st.markdown(f"<p style='text-align: center;'>Page {st.session_state.faq_page + 1} of {total_pages}</p>", unsafe_allow_html=True)
    with col_next:
        if st.button("Next ▶", use_container_width=True) and st.session_state.faq_page < total_pages - 1:
            st.session_state.faq_page += 1
    
    st.divider()
    
    # Display FAQs for current page
    start_idx = st.session_state.faq_page * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    for page_pos, global_idx in enumerate(filtered_indices[start_idx:end_idx], 1):
        col_expand, col_bookmark = st.columns([5, 1])
        
        with col_expand:
            with st.expander(f"Q: {questions[global_idx][:70]}..."):
                st.write(answers[global_idx])
                st.caption(f"Category: {categories[global_idx]}")
        
        with col_bookmark:
            if st.button("⭐", key=f"faq_bookmark_{global_idx}"):
                if add_bookmark(questions[global_idx], answers[global_idx]):
                    st.success("Bookmarked!")
                else:
                    st.info("Already bookmarked")

# -------- ADVANCED SEARCH --------
elif nav_choice == "🔍 Advanced Search":
    st.subheader("🔍 Advanced Search")
    
    search_type = st.radio("Search type:", ["By Keywords", "By Category", "By Similarity"])
    
    if search_type == "By Keywords":
        keyword = st.text_input("Enter keywords")
        min_words = st.slider("Minimum matching words:", 1, 5, 1)
        
        if keyword:
            results = []
            keywords = keyword.lower().split()
            for i, q in enumerate(questions):
                match_count = sum(1 for kw in keywords if kw in q.lower())
                if match_count >= min_words:
                    results.append((i, match_count))
            
            results = sorted(results, key=lambda x: x[1], reverse=True)[:20]
            
            if results:
                st.success(f"Found {len(results)} matching questions")
                for global_idx, matches in results:
                    with st.expander(f"Q: {questions[global_idx][:70]}... ({matches} matches)"):
                        st.write(answers[global_idx])
                        st.caption(f"Category: {categories[global_idx]}")
            else:
                st.warning("No matches found")
    
    elif search_type == "By Category":
        selected_cats = st.multiselect("Select categories:", unique_categories)
        if selected_cats:
            for cat in selected_cats:
                st.markdown(f"### {cat}")
                cat_items = [(i, q) for i, (q, c) in enumerate(zip(questions, categories)) if c == cat][:10]
                for idx, q in cat_items:
                    with st.expander(f"Q: {q[:70]}..."):
                        st.write(answers[idx])
    
    elif search_type == "By Similarity":
        ref_question = st.text_area("Enter a reference question:")
        num_similar = st.slider("Number of similar questions:", 1, 20, 5)
        
        if ref_question:
            query_vec = model.encode([ref_question])
            scores = cosine_similarity(query_vec, question_embeddings)[0]
            top_indices = np.argsort(scores)[::-1][:num_similar]
            
            for rank, idx in enumerate(top_indices, 1):
                similarity = scores[idx]
                with st.expander(f"#{rank} - {questions[idx][:70]}... ({similarity:.0%} similar)"):
                    st.write(answers[idx])
                    st.caption(f"Category: {categories[idx]}")

# -------- ANALYTICS --------
elif nav_choice == "📊 Analytics":
    st.subheader("📊 Analytics Dashboard")
    
    analytics = get_analytics()
    
    if analytics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Searches", analytics['total_searches'])
        with col2:
            st.metric("Avg Confidence", f"{analytics['avg_confidence']:.0%}")
        with col3:
            st.metric("Bookmarks", len(st.session_state.bookmarks))
        with col4:
            st.metric("Feedback Count", analytics['feedback_count'])
        
        st.divider()
        
        # Search history
        if st.session_state.search_history:
            st.markdown("### 📜 Recent Searches")
            df_history = pd.DataFrame(st.session_state.search_history[-10:])
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp']).dt.strftime('%H:%M:%S')
            st.dataframe(df_history[['timestamp', 'query', 'confidence']], use_container_width=True)
        
        # Top search terms
        if analytics['top_words']:
            st.markdown("### 🔤 Top Search Terms")
            terms_data = {'Term': [w for w, _ in analytics['top_words']], 'Count': [c for _, c in analytics['top_words']]}
            st.bar_chart(pd.DataFrame(terms_data).set_index('Term'))
        
        # Feedback statistics
        if st.session_state.feedback_data:
            st.markdown("### ⭐ Feedback Statistics")
            ratings = [f['rating'] for f in st.session_state.feedback_data]
            rating_counts = Counter(ratings)
            st.bar_chart(pd.DataFrame(list(rating_counts.items()), columns=['Rating', 'Count']).set_index('Rating'))
    else:
        st.info("No analytics data yet. Start using the app to see statistics!")

# -------- BOOKMARKS --------
elif nav_choice == "⭐ Bookmarks":
    st.subheader("⭐ My Bookmarks")
    
    if st.session_state.bookmarks:
        for i, bookmark in enumerate(st.session_state.bookmarks):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                with st.expander(f"Q: {bookmark['question'][:70]}..."):
                    st.write(bookmark['answer'])
                    st.caption(f"Saved on: {bookmark['timestamp'][:10]}")
            
            with col2:
                if st.button("🗑️", key=f"remove_bookmark_{i}"):
                    remove_bookmark(bookmark['question'])
                    st.rerun()
    else:
        st.info("No bookmarks yet. Click the bookmark button when viewing answers!")

# -------- FEEDBACK --------
elif nav_choice == "📝 Feedback":
    st.subheader("📝 Provide Feedback")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        feedback_query = st.text_input("Question you're rating (optional):")
        feedback_comment = st.text_area("Your feedback or suggestion:")
    
    with col2:
        feedback_rating = st.radio("Rate your experience:", [1, 2, 3, 4, 5])
    
    if st.button("Submit Feedback", type="primary", use_container_width=True):
        if feedback_comment.strip():
            add_feedback(feedback_query, feedback_rating, feedback_comment)
            st.success("Thank you for your feedback! 🙏")
        else:
            st.warning("Please enter your feedback")
    
    # Display feedback summary
    if st.session_state.feedback_data:
        st.divider()
        st.markdown("### 📊 Feedback Summary")
        ratings = [f['rating'] for f in st.session_state.feedback_data]
        avg = np.mean(ratings)
        st.metric("Average Rating", f"{avg:.1f}/5.0", delta=f"{len(ratings)} responses")

# -------- QUICK TIPS --------
elif nav_choice == "⚡ Quick Tips":
    st.subheader("⚡ Quick Tips & Tricks")
    
    tips = [
        ("🎯 Specific Questions", "Ask specific questions for better results"),
        ("🔑 Use Keywords", "Include relevant keywords from university policies"),
        ("📂 Browse by Category", "Use FAQ browser to explore topics"),
        ("⭐ Save Answers", "Bookmark important answers for quick access"),
        ("📊 Check Analytics", "View your search trends and patterns"),
        ("💬 Give Feedback", "Help us improve by rating answers"),
        ("🔍 Similar Questions", "Use advanced search to find similar topics"),
        ("🌙 Dark Mode", "Enable dark mode for comfortable reading"),
    ]
    
    cols = st.columns(2)
    for i, (title, desc) in enumerate(tips):
        with cols[i % 2]:
            st.info(f"{title}\n\n{desc}")

# -------- ADMIN PANEL --------
elif nav_choice == "🔐 Admin Panel":
    st.subheader("🔐 Admin Panel")
    
    # Password protection
    if not st.session_state.admin_mode:
        password = st.text_input("Enter admin password:", type="password")
        if st.button("Login"):
            if password == st.session_state.admin_password:
                st.session_state.admin_mode = True
                st.rerun()
            else:
                st.error("Invalid password")
    else:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.success("✅ Admin Mode Active")
        with col2:
            if st.button("Logout"):
                st.session_state.admin_mode = False
                st.rerun()
        
        st.divider()
        
        admin_tab1, admin_tab2, admin_tab3, admin_tab4 = st.tabs(
            ["Add Q&A", "Manage Q&A", "View Feedback", "Export Data"]
        )
        
        # Add Q&A
        with admin_tab1:
            st.markdown("### Add New Q&A Pair")
            new_question = st.text_input("Question:")
            new_answer = st.text_area("Answer:")
            new_category = st.selectbox("Category:", unique_categories + ["New Category"], key="admin_cat")
            
            if new_category == "New Category":
                new_category = st.text_input("Enter new category name:")
            
            if st.button("Add Q&A Pair", type="primary"):
                if new_question and new_answer and new_category:
                    st.session_state.custom_qa_pairs.append({
                        'question': new_question,
                        'answer': new_answer,
                        'category': new_category,
                        'added_on': datetime.now().isoformat()
                    })
                    save_persistent_data()
                    st.success("Q&A pair added successfully!")
                else:
                    st.warning("Please fill all fields")
        
        # Manage Q&A
        with admin_tab2:
            st.markdown("### Custom Q&A Pairs")
            if st.session_state.custom_qa_pairs:
                for i, qa in enumerate(st.session_state.custom_qa_pairs):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        with st.expander(f"Q: {qa['question'][:60]}..."):
                            st.write(f"**Answer:** {qa['answer']}")
                            st.caption(f"Category: {qa['category']}")
                    with col2:
                        if st.button("🗑️", key=f"admin_delete_{i}"):
                            st.session_state.custom_qa_pairs.pop(i)
                            save_persistent_data()
                            st.rerun()
            else:
                st.info("No custom Q&A pairs yet")
        
        # View Feedback
        with admin_tab3:
            st.markdown("### User Feedback")
            if st.session_state.feedback_data:
                df_feedback = pd.DataFrame(st.session_state.feedback_data)
                st.dataframe(df_feedback, use_container_width=True)
                
                if st.button("Clear All Feedback"):
                    st.session_state.feedback_data = []
                    save_persistent_data()
                    st.rerun()
            else:
                st.info("No feedback yet")
        
        # Export Data
        with admin_tab4:
            st.markdown("### Export Data")
            
            export_format = st.selectbox("Format:", ["JSON", "CSV"])
            
            data_to_export = {
                'bookmarks': st.session_state.bookmarks,
                'feedback': st.session_state.feedback_data,
                'custom_qa': st.session_state.custom_qa_pairs,
                'search_history': st.session_state.search_history,
                'exported_on': datetime.now().isoformat()
            }
            
            if export_format == "JSON":
                export_data = json.dumps(data_to_export, indent=2)
                st.download_button(
                    "Download JSON",
                    export_data,
                    "uniassist_export.json",
                    "application/json"
                )
            else:
                # CSV export (flattened)
                export_df = pd.json_normalize(st.session_state.feedback_data)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "uniassist_feedback.csv",
                    "text/csv"
                )

# ============ FOOTER ============
st.divider()
st.markdown(
    "<div class='footer'>© 2026 UniAssist | Enhanced Edition | Academic Project & Research Prototype</div>",
    unsafe_allow_html=True
)
