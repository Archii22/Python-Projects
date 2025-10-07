🧠 AI-Driven Data Insights Dashboard

An intelligent data analysis dashboard built with Streamlit, designed to automatically detect trends, anomalies, and generate AI-powered narrative insights from any uploaded dataset.
It’s your personal data analyst — powered by Python and AI.

✨ Features

✅ Automatic Trend Detection – Identifies increasing or decreasing patterns in numerical data.
✅ Anomaly Detection – Spots unusual data points or outliers in your dataset.
✅ Ranked Insights – Ranks insights by impact × confidence to highlight what matters most.
✅ AI-Powered Summaries – Generates narrative explanations for patterns and anomalies.
✅ Interactive Visuals – Explore your data using Plotly charts and Streamlit tabs.
✅ Fast & Cached – Optimized with @st.cache_data to run analysis under 10 seconds.
✅ Exportable Reports – Convert results to downloadable PDF summaries.

🧰 Tech Stack
Category	Tools Used
Frontend/UI	Streamlit
Data Handling	Pandas, NumPy
Visualization	Plotly Express
Machine Learning	scikit-learn
AI/LLM	OpenAI API
Database (optional)	DuckDB
Language	Python 3.11+
⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/Archii22/Python-Projects.git
cd "Python-Projects/project AI-Driven Data Insights Dashboard"

2️⃣ Create and Activate a Virtual Environment
python -m venv venv
venv\Scripts\activate     # For Windows
# OR
source venv/bin/activate  # For macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Your API Key

Create a file named .env in the project root and add:

OPENAI_API_KEY=your_api_key_here

5️⃣ Run the App
streamlit run app/streamlit_app.py


Then open your browser at 👉 http://localhost:8501

🧪 How It Works

Upload any CSV dataset (UTF-8 format).

The app automatically detects numeric columns.

It finds trends, anomalies, and patterns.

Insights are ranked by importance (impact × confidence).

AI generates a natural-language narrative summary.
