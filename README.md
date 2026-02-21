# UniAssist India

UniAssist India is a Streamlit app for academic and internship guidance with a strict dataset-first retrieval architecture.

## Core Logic (Preserved)

The answer pipeline is unchanged:
1. Load Q&A from `UniAssist_training_data.csv`
2. Generate embeddings with `all-MiniLM-L6-v2`
3. Compute cosine similarity
4. Apply threshold gating with safe fallback

No free-form answer generation is used.

## Role Portals (Login Required)

The app now has separate role sections, each with its own login gate:
- Student
- Teacher
- Parent
- Developer Admin

Demo credentials:
- `student_demo / student123`
- `teacher_demo / teacher123`
- `parent_demo / parent123`
- `admin_demo / admin123`

## Public Section

- `Feedback & Ratings` tab is visible to everyone.
- Global average rating is also visible in the sidebar.
- Feedback data can be downloaded as CSV.

## UI Upgrade

The interface has been polished for a smoother, premium feel:
- Improved contrast handling for all text inputs and text areas
- Safer field colors (input text no longer blends with background)
- Upgraded cards, tabs, forms, and buttons with consistent visual hierarchy
- Subtle motion/hover transitions for smoother interaction feedback
- Better readability with cleaner spacing and typography

## Universities

University names are anonymized and normalized to:
- `University 1`
- `University 2`
- `University 3`

## Feature Modules

- Dataset-grounded Assistant (confidence, trust score, citations, escalation)
- Student Toolkit (attendance/CGPA what-if, checklist planner, matcher)
- Teacher Desk (publish verified sources)
- Parent Overview (read-only snapshot + alerts)
- Developer Admin Console (review queue, source uploads, policy change detector, quality view)
- Analytics (query metrics, ticket queue, mini evaluation)
- Feedback & Ratings (rating form + export)

## Auto-Created Data Files

Stored under `data/`:
- `feedback.csv`
- `query_logs.csv`
- `alerts.csv`
- `tickets.csv`

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- This tool provides guidance only; official circulars remain authoritative.
- Author and copyright are shown in-app.
