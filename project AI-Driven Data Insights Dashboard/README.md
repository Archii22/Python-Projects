# 📊 AI-Driven Data Insights Dashboard

An **interactive Streamlit application** that empowers users to **analyze uploaded datasets** (CSV or Excel), detect **trends**, identify **anomalies**, rank insights by **impact × confidence**, and export results in multiple formats — including JSON, Markdown, and PDF.

---

# ✨ Key Features

## ✅ 1. Upload & Preview Data
- Upload **CSV or Excel** files (UTF-8 compatible).  
- Automatic **detection of data structure and types**.  
- Display **data summary, columns, and sample rows**.

## ✅ 2. AI-Powered Insights
- Detects **trends**: increasing, decreasing, or stable.  
- Identifies **anomalies** using statistical methods.  
- **Ranks insights** based on **impact × confidence**.  
- Generates **AI-driven narrative summaries** for quick understanding.

## ✅ 3. Interactive Visualizations
- Built with **Plotly** for dynamic trend and anomaly charts.  
- Optimized for datasets up to **1000 rows** for faster rendering.  

## ✅ 4. Natural Language Queries (NLQ)
- *(Coming Soon)* Ask questions like:  
  > “Which product category grew fastest?”  

## ✅ 5. Export Options
- Download results as **CSV, JSON, Markdown, or PDF**.  
- Auto-generate **reports with summaries and charts**.  

## ✅ 6. Performance & Safety
- Uses `@st.cache_data` for **caching and faster performance**.  
- Restricts **unsafe SQL queries** (only `SELECT` allowed).  
- Gracefully handles **missing or invalid data**.  
- Optimized to run within **~10 seconds** for medium datasets.

## ✅ 7. 🧪 Running Tests
This project includes automated tests using **pytest**, including:

- `test_insights_nlq.py` – tests the Natural Language Query (NLQ) functionality.

## Run Tests
Make sure your virtual environment is activated, then run:
#### pytest

### Command to run:
pytest tests/test_insights_nlq.py

---

# ⚙️ Setup Instructions

## Clone the Repository

git clone https://github.com/Archii22/Python-Projects.git
cd "Python-Projects/project AI-Driven Data Insights Dashboard"

---

#  Create and Activate a Virtual Environment

## Windows
python -m venv venv
venv\Scripts\activate

## macOS/Linux
python -m venv venv
source venv/bin/activate

---

## Install Dependencies
pip install -r requirements.txt

---

## Add Your OpenAI API Key
OPENAI_API_KEY=your_api_key_here

---

## Run the Application
streamlit run app/streamlit_app.py

---

# 📸 Demo Screenshots
### Data Upload & Preview
## 🖼️ App Preview

![BEFORE UPLOAD PREVIEW](demo%20page/uploadpreview.jpeg)

### Insights & Ranking Tab
### Full Dashboard in Action
