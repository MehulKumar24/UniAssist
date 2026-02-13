# 🎓 UniAssist

**Academic Guidance Assistant** – A fast, lightweight, retrieval-based Q&A system for university and internship queries.

---

## 📍 Overview

UniAssist is a controlled retrieval system designed to provide reliable answers to academic questions without hallucinations. It uses semantic similarity matching to find the most relevant answer from a curated dataset of 1075+ Q&A pairs.

**Key Focus:**
- ✨ Fast semantic search
- 📖 Retrieval-based (no generation)
- 🎯 Scope-controlled (academic only)
- 🐎 Lightweight & efficient (~384 lines)
- 📦 Zero bloat – only essential features

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/MehulKumar24/UniAssist.git
cd UniAssist

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

Access at: **http://localhost:8501**

---

## 🎯 Features

### 🏠 Home Page
- Semantic Q&A search with confidence scores
- Related questions displayed
- Real-time bookmark & review

### 📚 Browse FAQ
- Category-based filtering
- Paginated browsing (5/10/20/50 per page)
- One-click bookmarking

### 🔍 Advanced Search
- **By Keywords** – Multi-word matching
- **By Category** – Browse specific topics
- **By Similarity** – Find related questions

### ⭐ Bookmarks
- Save important Q&A pairs
- Persistent storage
- One-click removal

### 📝 Feedback
- Rate answers (1-5 stars)
- Leave comments
- View feedback summary

### 🔐 Admin Panel
- Add/manage custom Q&A pairs
- View all feedback
- Export data (JSON/CSV)
- Password-protected access

---

## 🏗️ Architecture

```
User Query
   ↓
Sentence-Embedding (all-MiniLM-L6-v2)
   ↓
Cosine Similarity Search
   ↓
Threshold Check (0.50)
   ↓
Return Answer OR Fallback
```

**Why This Design?**
- Predictable, explainable behavior
- No hallucinations
- Fast inference
- Low computational cost

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Code Size | 384 lines |
| Q&A Pairs | 1075+ |
| Search Latency | <1s |
| Memory | ~200MB |
| Categories | 24 |

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit** – Web framework
- **Sentence-Transformers** – Semantic embeddings (all-MiniLM-L6-v2)
- **Scikit-learn** – Cosine similarity
- **Pandas/NumPy** – Data handling
- **JSON** – Persistent storage

---

## 📦 Removed Features

Streamlined for performance:
- ❌ Dark/Light theme toggle
- ❌ Analytics dashboard
- ❌ Text-to-speech
- ❌ PDF export
- ❌ Multi-language support
- ❌ Quick tips page

**Result:** 47% code reduction (721 → 384 lines)

---

## 📁 Project Structure

```
UniAssist/
├── app.py                          # Main application (384 lines)
├── UniAssist_training_data.csv     # Q&A dataset (1075 pairs)
├── requirements.txt                # Dependencies
├── LICENSE                         # Apache 2.0
├── README.md                       # This file
│
├── 01_data_exploration.ipynb       # Data analysis
├── 02_retrieval_system.ipynb       # Semantic search
├── 03_safety_and_scope_control.ipynb # Scope control
└── 04_paraphrasing.ipynb           # ML training reference
```

---

## 💾 Data Persistence

- **uniassist_data.json** – Stores:
  - Bookmarks
  - Feedback entries
  - Custom Q&A pairs

Auto-saved on every action.

---

## 🔐 Security

- **Admin Password:** `admin123` (change in code)
- **Rate Limiting:** 100 queries per session
- **No External API Calls**
- **Local Storage Only**

---

## ⚙️ Configuration

Edit `app.py` to customize:

```python
THRESHOLD = 0.50              # Similarity threshold
RATE_LIMIT = 100             # Queries per session
'admin_password': "admin123" # Admin password
```

---

## 🌐 Deployment

### Streamlit Cloud
```bash
streamlit run app.py --server.port 8501
```

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
```

---

## 📚 Dataset

**UniAssist_training_data.csv**
- 1075 Q&A pairs
- 24 academic categories
- Manual curation
- Verified answers

⚠️ **Dataset Use Policy:**
- Custom-created and manually curated
- Attribution required for use
- Unauthorized redistribution discouraged

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No answer found" | Update `THRESHOLD` (default 0.50) |
| Slow search | Restart Streamlit session |
| Missing data | Check `uniassist_data.json` exists |
| Admin won't login | Verify password in code |

---

## 📈 Development Notebooks

1. **01_data_exploration.ipynb** – Dataset creation & validation
2. **02_retrieval_system.ipynb** – Semantic matching evaluation
3. **03_safety_and_scope_control.ipynb** – Threshold tuning & fallbacks
4. **04_paraphrasing.ipynb** – ML reference (not used in production)

---

## 📜 License

- **Code:** Apache License 2.0
- **Dataset:** Custom-curated (attribution required)

---

## 👤 Author

**Mehul Kumar**  
B.Tech (1st Year) | South Asian University, New Delhi

---

## 📝 Citation

If using UniAssist dataset or code:

```
@software{uniassist2026,
  title = {UniAssist: Academic Guidance Assistant},
  author = {Kumar, Mehul},
  year = {2026},
  url = {https://github.com/MehulKumar24/UniAssist}
}
```

---

## ✅ Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Feb 2026 | Initial release |
| 1.1 | Feb 2026 | Streamlined to 384 lines |
| 1.2 | Feb 2026 | Removed non-essential features |

---

**Status:** Active Development | Last Updated: February 13, 2026
