
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
import html

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
    'faq_page': 0,
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
    },
    'es': {
        'title': '🎓 UniAssist',
        'subtitle': 'Asistente de Orientación Académica - Edición Mejorada',
        'ask_question': 'Haz tu pregunta',
        'get_answer': 'Obtener Respuesta',
        'answer': 'Respuesta',
        'confidence': 'Puntuación de Confianza',
        'related': 'Preguntas Relacionadas',
        'search_history': 'Historial de Búsqueda',
        'bookmarks': 'Marcadores',
        'feedback': 'Proporcionar Retroalimentación',
        'home': 'Inicio',
        'browse': 'Explorar FAQ',
        'search': 'Búsqueda Avanzada',
        'analytics': 'Panel de Análisis',
        'admin': 'Panel de Administrador',
    }
}

# ============ CUSTOM CSS ============
def get_theme_css(theme):
    if theme == 'dark':
        return """
<style>
body { background-color: #1a1a1a; color: #fff; }
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
.confidence-high { color: #2ecc71; font-weight: bold; }
.confidence-medium { color: #f39c12; font-weight: bold; }
.confidence-low { color: #e74c3c; font-weight: bold; }
.footer { font-size: 12px; color: #666; text-align: center; margin-top: 50px; }
</style>
"""
    else:
        return """
<style>
body { background-color: #ffffff; color: #000; }
.main-title {
    font-size: 40px;
    font-weight: 700;
    color: #1f4ed8;
    text-align: center;
    margin-bottom: 10px;
}
.sub-title {
    font-size: 16px;
    color: #333;
    text-align: center;
    margin-bottom: 20px;
}
.answer-box {
    background-color: #e6f4ff;
    padding: 20px;
    border-radius: 10px;
    border-left: 6px solid #1f4ed8;
    color: #000;
    margin: 15px 0;
}
.confidence-high { color: #2ecc71; font-weight: bold; }
.confidence-medium { color: #f39c12; font-weight: bold; }
.confidence-low { color: #e74c3c; font-weight: bold; }
.footer { font-size: 12px; color: #555; text-align: center; margin-top: 50px; }
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
        except Exception as e:
            st.warning(f"Could not load persistent data: {e}")

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
    try:
        df = pd.read_csv("UniAssist_training_data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    """Load embedding model"""
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load base data
qa_frame = load_data()
if not qa_frame.empty:
    questions = qa_frame["question"].astype(str).tolist()
    answers = qa_frame["answer"].astype(str).tolist()
    categories = qa_frame["category_name"].astype(str).tolist()
else:
    questions, answers, categories = [], [], []

# Add custom Q&A pairs
if st.session_state.custom_qa_pairs:
    for qa in st.session_state.custom_qa_pairs:
        try:
            questions.append(qa.get('question', ''))
            answers.append(qa.get('answer', ''))
            categories.append(qa.get('category', 'Custom'))
        except:
            pass

model = load_model()
if model and questions:
    try:
        question_embeddings = model.encode(questions)
    except:
        question_embeddings = np.array([])
else:
    question_embeddings = np.array([])

# Get unique categories
unique_categories = sorted(set(cat for cat in categories if cat))

# ============ CONSTANTS & SETTINGS ============
SIMILARITY_THRESHOLD = 0.50
SAFE_FALLBACK_MESSAGE = (
    "I'm sorry, I don't have reliable information on this topic. "
    "UniAssist currently handles academic and internship-related queries only. "
    "Try using advanced search or browsing the FAQ."
)
RATE_LIMIT = 100

# ============ UTILITY FUNCTIONS ============

def get_lang_code():
    """Get language code from session state safely"""
    raw_lang = st.session_state.get('language', 'English')
    if raw_lang in LANGUAGES:
        return LANGUAGES[raw_lang]
    elif raw_lang in TRANSLATIONS:
        return raw_lang
    return 'en'

def sanitize_input(text):
    """Sanitize user input"""
    return str(text).strip()[:500]

def escape_html(text):
    """Escape HTML special characters"""
    return html.escape(str(text))

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
    try:
        if not check_rate_limit():
            return None, None, []
        
        if not model or not question_embeddings.size:
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
        for i, idx in enumerate(top_indices[1:min(4, len(top_indices))]):
            score_val = float(top_scores[i + 1])
            if score_val >= (SIMILARITY_THRESHOLD - 0.05):
                related.append({
                    'question': questions[idx],
                    'answer': answers[idx],
                    'score': score_val
                })
        
        return answers[best_index], float(best_score), related
    except Exception as e:
        st.error(f"Error retrieving answer: {e}")
        return None, None, []

def add_to_history(query, answer, confidence):
    """Add query to search history"""
    try:
        st.session_state.search_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': answer[:100] + "..." if len(answer) > 100 else answer,
            'confidence': confidence
        })
    except:
        pass

def add_bookmark(question, answer):
    """Add bookmark"""
    try:
        if not any(b['question'] == question for b in st.session_state.bookmarks):
            st.session_state.bookmarks.append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': answer
            })
            save_persistent_data()
            return True
        return False
    except:
        return False

def remove_bookmark(question):
    """Remove bookmark"""
    try:
        st.session_state.bookmarks = [b for b in st.session_state.bookmarks if b['question'] != question]
        save_persistent_data()
    except:
        pass

def add_feedback(query, rating, comment=""):
    """Store feedback"""
    try:
        st.session_state.feedback_data.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'rating': int(rating),
            'comment': comment
        })
        save_persistent_data()
    except:
        pass

def get_analytics():
    """Generate analytics"""
    try:
        if not st.session_state.search_history:
            return None
        
        queries = [h['query'] for h in st.session_state.search_history]
        total_searches = len(queries)
        confidences = [h['confidence'] for h in st.session_state.search_history if isinstance(h.get('confidence'), (int, float))]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Most searched words
        all_words = ' '.join(queries).lower().split()
        word_freq = Counter(all_words)
        
        # Filter common words
        common_words = {'what', 'is', 'the', 'a', 'and', 'or', 'for', 'to', 'in', 'of', 'how', 'can', 'i', 'about'}
        word_freq = {w: c for w, c in word_freq.items() if w not in common_words and len(w) > 2}
        
        # Feedback stats
        feedback = st.session_state.feedback_data
        ratings = [f['rating'] for f in feedback if isinstance(f.get('rating'), (int, float))]
        avg_rating = np.mean(ratings) if ratings else 0
        
        return {
            'total_searches': total_searches,
            'avg_confidence': avg_confidence,
            'top_words': Counter(word_freq).most_common(5),
            'feedback_count': len(feedback),
            'avg_rating': avg_rating
        }
    except Exception as e:
        st.error(f"Error generating analytics: {e}")
        return None

def export_to_pdf(question, answer):
    """Export answer to PDF"""
    if not PDF_AVAILABLE:
        return None
    
    try:
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        
        p.setFont("Helvetica-Bold", 16)
        p.drawString(50, 750, "UniAssist - Answer Export")
        
        p.setFont("Helvetica", 10)
        p.drawString(50, 720, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, 690, "Question:")
        p.setFont("Helvetica", 10)
        y = 670
        for line in question.split('\n')[:10]:
            if y < 100:
                p.showPage()
                y = 750
            p.drawString(70, y, line[:80])
            y -= 15
        
        y -= 10
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, y, "Answer:")
        p.setFont("Helvetica", 10)
        y -= 20
        for line in answer.split('\n')[:20]:
            if y < 100:
                p.showPage()
                y = 750
            p.drawString(70, y, line[:80])
            y -= 15
        
        p.setFont("Helvetica-Oblique", 8)
        p.drawString(50, 30, "UniAssist © 2026 | Academic Guidance Assistant")
        
        p.save()
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"PDF export error: {e}")
        return None

def text_to_speech(text, lang_code='en'):
    """Convert text to speech"""
    if not GTTS_AVAILABLE:
        return None
    
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio = BytesIO()
        tts.write_to_fp(audio)
        audio.seek(0)
        return audio
    except Exception as e:
        st.warning(f"Text-to-speech error: {e}")
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
        lang = st.selectbox("🌐 Language", options=list(LANGUAGES.keys()), index=list(LANGUAGES.keys()).index(st.session_state.language) if st.session_state.language in LANGUAGES else 0)
        if lang != st.session_state.language:
            st.session_state.language = lang
            st.rerun()
    
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

# ============ MAIN HEADER ============
lang_code = get_lang_code()
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
            placeholder="e.g., What is the minimum attendance requirement?",
            key="home_query"
        )
        
        col_search, col_example = st.columns([3, 1])
        with col_search:
            ask_button = st.button("🔍 Get Answer", use_container_width=True, key="home_search")
        with col_example:
            if st.button("? Example", use_container_width=True, key="home_example"):
                user_query = "What is the minimum attendance requirement?"
                st.rerun()
        
        if ask_button and user_query.strip():
            user_query = sanitize_input(user_query)
            
            with st.spinner("🔍 Searching for answer..."):
                answer, confidence, related = get_answer_with_confidence(user_query)
            
            if answer:
                add_to_history(user_query, answer, confidence)
                
                # Display answer
                st.markdown("### 📘 Answer")
                st.markdown(f"<div class='answer-box'>{escape_html(answer)}</div>", unsafe_allow_html=True)
                
                # Confidence score
                color_class, confidence_text = get_confidence_color(confidence)
                st.markdown(
                    f"<p style='text-align: center;'><span class='{color_class}'>{confidence_text}: {confidence:.1%}</span></p>",
                    unsafe_allow_html=True
                )
                
                # Action buttons
                action_col1, action_col2, action_col3, action_col4, action_col5 = st.columns(5)
                
                with action_col1:
                    if st.button("⭐ Bookmark", use_container_width=True, key="bookmark_home"):
                        if add_bookmark(user_query, answer):
                            st.success("Added to bookmarks!")
                        else:
                            st.info("Already bookmarked")
                
                with action_col2:
                    if st.button("📋 Copy", use_container_width=True, key="copy_home"):
                        st.info("Use Ctrl+C to copy")
                
                with action_col3:
                    if GTTS_AVAILABLE and st.button("🔊 Speak", use_container_width=True, key="speak_home"):
                        lang_code = get_lang_code()
                        audio = text_to_speech(answer, lang_code)
                        if audio:
                            st.audio(audio, format="audio/mp3")
                
                with action_col4:
                    if PDF_AVAILABLE and st.button("📄 PDF", use_container_width=True, key="pdf_home"):
                        pdf = export_to_pdf(user_query, answer)
                        if pdf:
                            st.download_button(
                                label="Download PDF",
                                data=pdf,
                                file_name="uniassist_answer.pdf",
                                mime="application/pdf"
                            )
                
                with action_col5:
                    if st.button("👍 Rate", use_container_width=True, key="rate_home"):
                        st.info("Use feedback section to rate")
                
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
            st.metric("Feedback", stats['feedback_count'])
        else:
            st.info("No data yet")

# -------- BROWSE FAQ --------
elif nav_choice == "📚 Browse FAQ":
    st.subheader("📚 Browse Frequently Asked Questions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_category = st.selectbox(
            "Filter by category:",
            options=["All Categories"] + unique_categories,
            key="faq_category"
        )
    
    with col2:
        items_per_page = st.selectbox("Items per page:", [5, 10, 20, 50], key="faq_items")
    
    # Filter Q&A
    if selected_category == "All Categories":
        filtered_indices = list(range(len(questions)))
    else:
        filtered_indices = [i for i, cat in enumerate(categories) if cat == selected_category]
    
    # Pagination
    total_items = len(filtered_indices)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    col_prev, col_page, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("◀ Previous", use_container_width=True, key="faq_prev"):
            if st.session_state.faq_page > 0:
                st.session_state.faq_page -= 1
                st.rerun()
    with col_page:
        st.markdown(f"<p style='text-align: center;'>Page {st.session_state.faq_page + 1} of {max(1, total_pages)}</p>", unsafe_allow_html=True)
    with col_next:
        if st.button("Next ▶", use_container_width=True, key="faq_next"):
            if st.session_state.faq_page < total_pages - 1:
                st.session_state.faq_page += 1
                st.rerun()
    
    st.divider()
    
    # Display FAQs for current page
    start_idx = st.session_state.faq_page * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    if total_items > 0:
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
    else:
        st.info("No questions in this category")

# -------- ADVANCED SEARCH --------
elif nav_choice == "🔍 Advanced Search":
    st.subheader("🔍 Advanced Search")
    
    search_type = st.radio("Search type:", ["By Keywords", "By Category", "By Similarity"], key="search_type")
    
    if search_type == "By Keywords":
        keyword = st.text_input("Enter keywords:", key="keyword_search")
        min_words = st.slider("Minimum matching words:", 1, 5, 1, key="min_words")
        
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
        selected_cats = st.multiselect("Select categories:", unique_categories, key="category_search")
        if selected_cats:
            for cat in selected_cats:
                st.markdown(f"### {cat}")
                cat_items = [(i, q) for i, (q, c) in enumerate(zip(questions, categories)) if c == cat][:10]
                for idx, q in cat_items:
                    with st.expander(f"Q: {q[:70]}..."):
                        st.write(answers[idx])
    
    elif search_type == "By Similarity":
        ref_question = st.text_area("Enter a reference question:", key="similarity_search")
        num_similar = st.slider("Number of similar questions:", 1, 20, 5, key="num_similar")
        
        if ref_question and model and question_embeddings.size:
            try:
                query_vec = model.encode([ref_question])
                scores = cosine_similarity(query_vec, question_embeddings)[0]
                top_indices = np.argsort(scores)[::-1][:num_similar]
                
                for rank, idx in enumerate(top_indices, 1):
                    similarity = scores[idx]
                    with st.expander(f"#{rank} - {questions[idx][:70]}... ({similarity:.0%} similar)"):
                        st.write(answers[idx])
                        st.caption(f"Category: {categories[idx]}")
            except Exception as e:
                st.error(f"Search error: {e}")

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
    else:
        st.info("No analytics data yet")

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
        st.info("No bookmarks yet")

# -------- FEEDBACK --------
elif nav_choice == "📝 Feedback":
    st.subheader("📝 Provide Feedback")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        feedback_query = st.text_input("Question you're rating (optional):", key="feedback_query")
        feedback_comment = st.text_area("Your feedback or suggestion:", key="feedback_comment")
    
    with col2:
        feedback_rating = st.radio("Rate your experience:", [1, 2, 3, 4, 5], key="feedback_rating")
    
    if st.button("Submit Feedback", type="primary", use_container_width=True):
        if feedback_comment.strip():
            add_feedback(feedback_query, feedback_rating, feedback_comment)
            st.success("Thank you for your feedback! 🙏")
            st.rerun()
        else:
            st.warning("Please enter your feedback")
    
    # Display feedback summary
    if st.session_state.feedback_data:
        st.divider()
        st.markdown("### 📊 Feedback Summary")
        ratings = [f['rating'] for f in st.session_state.feedback_data if isinstance(f.get('rating'), int)]
        if ratings:
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
        password = st.text_input("Enter admin password:", type="password", key="admin_pass")
        if st.button("Login", key="admin_login"):
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
            if st.button("Logout", key="admin_logout"):
                st.session_state.admin_mode = False
                st.rerun()
        
        st.divider()
        
        admin_tab1, admin_tab2, admin_tab3, admin_tab4 = st.tabs(
            ["Add Q&A", "Manage Q&A", "View Feedback", "Export Data"]
        )
        
        # Add Q&A
        with admin_tab1:
            st.markdown("### Add New Q&A Pair")
            new_question = st.text_input("Question:", key="new_question")
            new_answer = st.text_area("Answer:", key="new_answer")
            new_category = st.selectbox("Category:", unique_categories + ["New Category"], key="admin_cat")
            
            if new_category == "New Category":
                new_category = st.text_input("Enter new category name:", key="new_cat_name")
            
            if st.button("Add Q&A Pair", type="primary", key="add_qa"):
                if new_question and new_answer and new_category:
                    st.session_state.custom_qa_pairs.append({
                        'question': new_question,
                        'answer': new_answer,
                        'category': new_category,
                        'added_on': datetime.now().isoformat()
                    })
                    save_persistent_data()
                    st.success("Q&A pair added successfully!")
                    st.rerun()
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
                
                if st.button("Clear All Feedback", key="clear_feedback"):
                    st.session_state.feedback_data = []
                    save_persistent_data()
                    st.rerun()
            else:
                st.info("No feedback yet")
        
        # Export Data
        with admin_tab4:
            st.markdown("### Export Data")
            
            export_format = st.selectbox("Format:", ["JSON", "CSV"], key="export_format")
            
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
                    "application/json",
                    key="download_json"
                )
            else:
                # CSV export (flattened)
                if st.session_state.feedback_data:
                    export_df = pd.json_normalize(st.session_state.feedback_data)
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "uniassist_feedback.csv",
                        "text/csv",
                        key="download_csv"
                    )
                else:
                    st.info("No feedback data to export")

# ============ FOOTER ============
st.divider()
st.markdown(
    "<div class='footer'>© 2026 UniAssist | Enhanced Edition | Academic Project & Research Prototype</div>",
    unsafe_allow_html=True
)
