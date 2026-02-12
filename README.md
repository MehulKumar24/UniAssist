🎓 UniAssist
Academic & Internship Guidance Assistant

UniAssist is a controlled, retrieval-based academic assistance system designed to answer university-related and internship-related queries accurately and safely.
The project emphasizes correctness, scope control, and explainability, avoiding the common pitfalls of unrestricted generative chatbots.



📍 Problem Statement

Students frequently face difficulty in accessing clear, consistent, and reliable information regarding:

Attendance policies

Internship eligibility and rules

Academic procedures

Examination and grading systems

Generic AI chatbots often:

Hallucinate answers

Provide out-of-scope information

Lack accountability in academic contexts

UniAssist addresses these challenges by grounding responses in a curated dataset and enforcing strict scope control.



🎯 Project Objectives

Build a safe academic assistant that answers only verified queries

Demonstrate understanding of semantic retrieval techniques

Avoid hallucinations using similarity thresholds and fallback logic

Deploy a real, usable web application

Maintain academic integrity and originality



🧠 System Architecture Overview

UniAssist follows a retrieval-first architecture, not a free-form generative model.

High-level flow:

User Query
   ↓
Sentence Embedding
   ↓
Similarity Matching (Cosine Similarity)
   ↓
Safety Threshold Check
   ↓
Retrieved Answer OR Safe Fallback

This design ensures:

Predictable behavior

Transparent logic

Reduced risk of incorrect answers



🗂️ Project Structure

UniAssist/
│
├── 01_data_exploration.ipynb
├── 02_retrieval_system.ipynb
├── 03_safety_and_scope_control.ipynb
├── 04_paraphrasing.ipynb
│
├── UniAssist_training_data.csv
├── app.py
├── requirements.txt
├── LICENSE
└── README.md



📘 Development Notebooks (Explanation)

🔹 Notebook 01 – Data Exploration & Validation

Manual creation of question–answer pairs

Categorization of academic and internship queries

Dataset consistency and formatting checks

Exploratory analysis of Q&A data distribution

Outcome:
A custom dataset (UniAssist_training_data.csv) used directly by the app.



🔹 Notebook 02 – Semantic Retrieval System

Sentence embeddings using SentenceTransformer (all-MiniLM-L6-v2)

Similarity computation using cosine similarity

Evaluation of retrieval quality and threshold tuning

Testing retrieval-based Q&A system

Outcome:
Reliable semantic matching between user queries and stored questions with confidence scoring.



🔹 Notebook 03 – Safety, Scope Control & System Architecture

Definition of in-scope vs out-of-scope queries

Similarity threshold tuning (0.50 baseline)

Polite fallback responses for unsupported questions

Rate limiting and user safety mechanisms

Outcome:
Prevention of hallucinated or unrelated answers with robust scope control.



🔹 Notebook 04 – Seq2Seq / Paraphrasing Model

Training a Seq2Seq model to demonstrate ML training workflow

Exploration of response paraphrasing techniques

Note:
The deployed system prioritizes retrieval-based reliability.
The Seq2Seq model is included as learning and enhancement evidence, not as the primary answer generator.



🧩 Core Technologies Used

Python

Streamlit – Web application framework

Sentence-Transformers – Semantic embeddings (all-MiniLM-L6-v2)

Scikit-learn – Similarity computation

Pandas / NumPy – Data handling

PyTorch – Model backend (via Sentence-Transformers)

gTTS – Text-to-speech functionality

ReportLab – PDF export capability



✨ Enhanced Web Application Features

The enhanced Streamlit app includes:

🏠 Home Page – Smart Q&A with confidence scoring

📚 Browse FAQ – Category-based filtering with pagination

🔍 Advanced Search – Multiple search modes (Keywords, Category, Similarity)

📊 Analytics Dashboard – Track searches, confidence trends, feedback statistics

⭐ Bookmarks – Save and organize important answers

📝 Feedback System – User feedback collection and analysis

⚡ Quick Tips – Helpful usage tips and best practices

🔐 Admin Panel – Add/Manage Q&A pairs, view feedback, export data

🌙 Dark/Light Mode – Theme toggle for comfortable reading

🌐 Multi-language Support – English, Hindi, Spanish

📄 PDF Export – Download answers as PDF documents

🔊 Text-to-Speech – Audio playback of answers (when available)

📊 Search History – Track all previous queries

🔖 Persistent Data – Bookmarks, feedback, and custom Q&A saved locally



🌐 Web Application (Deployment & Installation)

Deployed using Streamlit

Local Installation

Clone the repository:
git clone https://github.com/MehulKumar24/UniAssist.git
cd UniAssist

Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app.py

Access via browser at: http://localhost:8501



Requirements

Python 3.8+

All packages listed in requirements.txt

Supported on Desktop, Tablet, and Mobile browsers



📱 Mobile Usage

UniAssist does not require native installation.

Steps:

Open the deployed app URL in a mobile browser

Use “Add to Home Screen”

Launch like a normal app



⚠️ Scope & Limitations

Responds only to academic and internship-related queries

Does not replace official university notifications

Answers are limited to the provided dataset

Internet connection required

These limitations are intentional to ensure safety and correctness.



📜 Copyright, License & Usage Policy

© 2026 Mehul Kumar. All rights reserved.

The source code of this project is licensed under the Apache License, Version 2.0.
Use, modification, and distribution of the code are permitted under the terms of the license, provided that proper attribution to the original author is maintained.
The software is provided “AS IS”, without warranties or conditions of any kind.



Dataset Ownership & Restrictions

The dataset used in this project (UniAssist_training_data.csv) is custom-created, manually curated, and authored by the project owner specifically for academic and demonstrative purposes.

The dataset is not autogenerated, scraped, or externally sourced

Unauthorized reuse, redistribution, or repackaging of the dataset—either in full or in part—without attribution is strongly discouraged

Any academic or derivative use must explicitly credit the original author



Academic Integrity Statement

This project was developed as part of academic learning and evaluation.
The system design, dataset structure, logic flow, and implementation choices reflect the author’s independent understanding, experimentation, and decision-making.

Any reuse of this work should:

Maintain academic honesty

Avoid misrepresentation of authorship

Respect institutional and ethical guidelines



Commercial & Derivative Use Notice

While the Apache License 2.0 permits commercial use of the codebase, any commercial or large-scale deployment of this system should:

Clearly disclose system limitations

Ensure responsible and ethical usage

Respect dataset ownership and attribution

The author assumes no liability for misuse or misinterpretation of outputs.



Final Note

This project is shared publicly for learning, transparency, and evaluation, not for uncredited replication.
Responsible use and proper attribution are expected and appreciated.



🚀 Future Scope & Enhancements

Potential future improvements include:

Multi-university support

Role-based access (student / faculty)

Feedback-based response refinement

Hybrid retrieval + generation architecture

API-based backend for commercial deployment



👤 Author

Mehul Kumar
B.Tech (1st Year)
South Asian University, New Delhi
IIT Madras, Chennai



✅ Final Remarks

UniAssist Enhanced was built with an emphasis on:

Understanding over automation

Safety over speculation

Structure over improvisation

User experience and accessibility

The project demonstrates not only technical implementation but also responsible system design, making it suitable for both academic evaluation and future real-world extension.

**Important:** Copying of dataset is not recommended. Codes may be referenced as they represent fundamental programming concepts. If the dataset is used without proper attribution, legal action may apply as required.

---

**Status:** Active Development | Last Updated: February 2026 | Version: Enhanced Edition 
