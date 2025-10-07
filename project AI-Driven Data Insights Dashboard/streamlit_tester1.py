import io
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Optional
import duckdb
import json
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
import plotly.express as px  # for plotting anomalies

# --------------------------
# Helper Functions
# --------------------------

# Load .env file
load_dotenv("password.env")

# Missing function: summarize_insights
def summarize_insights(insights: list) -> str:
    """
    Convert a list of insight strings into a bullet-point narrative summary.
    """
    if not insights:
        return "No insights available."
    return "\n".join([f"• {i}" for i in insights])

# --------------------------
# Streamlit page setup
# --------------------------
st.set_page_config(page_title="File Uploader — CSV / Excel", layout="wide")
st.title("📥 File uploader — CSV / Excel")
st.markdown("Upload a CSV or Excel file, preview it, run light cleaning, and download the cleaned CSV.")

# Get API key from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------
# CSV Reader with delimiter sniffing
# --------------------------
def read_csv_with_sniff(file_bytes: bytes, encoding: str = "utf-8") -> pd.DataFrame:
    try:
        text = file_bytes.decode(encoding)
    except Exception:
        # fallback to binary read if decoding fails
        return pd.read_csv(io.BytesIO(file_bytes))

    delimiters = [",", "\t", ";", "|"]
    best_df = None
    best_cols = 0
    for d in delimiters:
        try:
            df = pd.read_csv(io.StringIO(text), sep=d)
            if df.shape[1] > best_cols:
                best_cols = df.shape[1]
                best_df = df
        except Exception:
            continue

    if best_df is not None:
        return best_df
    return pd.read_csv(io.StringIO(text))

# --------------------------
# Excel Reader
# --------------------------
def read_excel(file_bytes: bytes, sheet_name: Optional[str] = None) -> dict:
    with io.BytesIO(file_bytes) as bio:
        xl = pd.ExcelFile(bio, engine="openpyxl")
        sheets = xl.sheet_names
        out = {}
        for s in sheets:
            out[s] = xl.parse(s)
        return out

# --------------------------
# Trend Detection Functions
# --------------------------
def detect_trend_simple(series: pd.Series) -> str:
    if not np.issubdtype(series.dtype, np.number):
        return "non-numeric"
    s = series.dropna()
    if len(s) < 2:
        return "insufficient data"
    if s.iloc[-1] > s.iloc[0]:
        return "increasing"
    elif s.iloc[-1] < s.iloc[0]:
        return "decreasing"
    else:
        return "stable"

def detect_trend_regression(series: pd.Series) -> str:
    if not np.issubdtype(series.dtype, np.number):
        return "non-numeric"
    s = series.dropna()
    if len(s) < 2:
        return "insufficient data"
    x = np.arange(len(s)).reshape(-1, 1)
    y = s.values.reshape(-1, 1)
    model = LinearRegression()
    try:
        model.fit(x, y)
        slope = float(model.coef_[0][0])
    except Exception:
        # fallback to simple check
        return detect_trend_simple(series)
    if slope > 0:
        return "increasing"
    elif slope < 0:
        return "decreasing"
    else:
        return "stable"
    
def detect_trends(df):
    trends={}
    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().sum()>0:
            continue
        x= np.arange(len(df))
        y=df[col].values
        corr= np.corrcoef(x,y)[0,1]
        if corr > 0.3:
            trends[col] = "increasing"
        elif corr < -0.3:
            trends[col] = "decreasing"
        else:
            trends[col] = "stable"
    return trends

# --------------------------
# Ranking Insight
# --------------------------
def rank_insights(insights: list) -> pd.DataFrame:
    """
    Rank insights by importance = impact * confidence.
    """
    df = pd.DataFrame(insights)
    if df.empty:
        return pd.DataFrame(columns=["insight", "impact", "confidence", "score"])

    df["impact"] = pd.to_numeric(df.get("impact", 0), errors="coerce").fillna(0)
    df["confidence"] = pd.to_numeric(df.get("confidence", 0), errors="coerce").fillna(0)
    df["impact_norm"] = (df["impact"] - df["impact"].min()) / (df["impact"].max() - df["impact"].min() + 1e-9)
    df["confidence_norm"] = (df["confidence"] - df["confidence"].min()) / (df["confidence"].max() - df["confidence"].min() + 1e-9)
    df["score"] = df["impact_norm"] * df["confidence_norm"]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df[["insight", "impact", "confidence", "score"]]

# --------------------------
# OpenAI
# --------------------------
def ask_openai(prompt: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data insights assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

print(ask_openai("Summarize this text in 2 lines."))

# --------------------------
# NLQ → SQL / Pandas
# --------------------------
def nlq_to_code(nl_query: str, engine:str = "pandas") -> str:
    prompt = f"""
    You are a Python assistant. The user has a dataset stored as:
    - Pandas DataFrame named `df` (if engine='pandas')
    - DuckDB table named `uploaded_data` (if engine='sql')
    Generate {engine} code to answer the following query. Only return code, no explainations.
    Query: "{nl_query}"
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that converts natural language to code."},
                {"role": "user", "content": prompt}
            ]
        )
        code= response.choices[0].message.content.strip()
        return code
    except Exception as e:
        return f"Error: {e}"

def execute_generated_code(code:str, engine:str = "pandas"):
    try:
        if engine == "pandas":
            local_vars = {"df": cleaned.copy(), "pd": pd, "np": np}
            exec(code, {}, local_vars)
            if "result" not in local_vars:
                return "⚠️ No `result` variable found in code."
            return local_vars.get("result", None)
        elif engine == "sql":
            result = con.execute(code).df()
            return result
    except Exception as e:
        return f"Execution error: {e}"

# --------------------------
# Sidebar upload
# --------------------------
st.sidebar.header("Upload options")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xls", "xlsx", "xlsb"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        sheets = read_excel(file_bytes)
        if len(sheets) == 1:
            df = list(sheets.values())[0]
        else:
            sheet_choice = st.sidebar.selectbox("Select sheet", options=list(sheets.keys()))
            df = sheets[sheet_choice]
    st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns from uploaded file.")
else:
    st.info("Upload a CSV or Excel file from the sidebar to get started.")
    st.stop()

# --------------------------
# Preview & Cleaning
# --------------------------
show_preview_rows = st.sidebar.number_input("Preview rows", min_value=1, max_value=1000, value=5)
parse_dates = st.sidebar.checkbox("Try to parse dates automatically", value=True)
st.sidebar.markdown("---")
clean_drop_na_cols = st.sidebar.checkbox("Drop columns with >N% missing", value=False)
na_threshold = None
if clean_drop_na_cols:
    na_threshold = st.sidebar.slider("Drop column if > N% missing", 0, 100, 80)
clean_drop_na_rows = st.sidebar.checkbox("Drop rows with any missing", value=False)
rename_columns = st.sidebar.checkbox("Rename columns (trim & lowercase)", value=False)

file_name = uploaded_file.name
st.sidebar.write(f"Selected file: {file_name}")
st.sidebar.write(f"File size: {len(file_bytes)} bytes")

# Read file with sniffing/parsing
try:
    if file_name.lower().endswith(".csv"):
        with st.spinner("Reading CSV..."):
            df = read_csv_with_sniff(file_bytes)
    else:
        with st.spinner("Reading Excel..."):
            sheets = read_excel(file_bytes)
            if len(sheets) == 1:
                df = list(sheets.values())[0]
            else:
                sheet_choice = st.sidebar.selectbox("Select sheet", options=list(sheets.keys()))
                df = sheets[sheet_choice]
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

if not isinstance(df, pd.DataFrame):
    st.error("Uploaded file did not return a DataFrame.")
    st.stop()
else:
    st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns from uploaded file.")

if parse_dates:
    with st.spinner("Parsing dates (best-effort)..."):
        for col in df.columns:
            try:
                converted = pd.to_datetime(df[col], errors="coerce", infer_datetime_format = True)
                non_null = converted.notna().sum()
                # require at least 3 parsed rows AND at least ~50% parsed to treat column as date
                if not_null >= max(3, int(0.5* len(df)))
            except Exception:
                pass

# --------------------------
# Show basic info
# --------------------------
st.subheader(f"Preview — {file_name}")
col1, col2 = st.columns([1, 2])
with col1:
    st.write("**Shape**")
    st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
    st.write("---")
    st.write("**Columns**")
    st.write(list(df.columns))
with col2:
    st.dataframe(df.head(show_preview_rows))

# --------------------------
# Cleaning
# --------------------------
cleaned = df.copy()
if clean_drop_na_cols and na_threshold is not None:
    thresh = int((100 - na_threshold) / 100.0 * len(cleaned))
    cleaned = cleaned.dropna(axis=1, thresh=thresh)
    st.write(f"Dropped columns with >{na_threshold}% missing")

if clean_drop_na_rows:
    cleaned = cleaned.dropna()
    st.write("Dropped rows with any missing values")

if rename_columns:
    cleaned.columns = [str(c).strip().lower().replace(" ", "_") for c in cleaned.columns]
    st.write("Normalized column names")

st.write("---")
st.write("**Preview cleaned data**")
st.dataframe(cleaned.head(show_preview_rows))

# --------------------------
# Schema detection
# --------------------------
st.subheader("🔎 Schema Detection")
dtypes_df = pd.DataFrame(df.dtypes, columns=["dtype"])
st.dataframe(dtypes_df)
missing_df = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(2)
})
st.dataframe(missing_df)
outlier_summary = {}
for col in df.select_dtypes(include=["number"]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_summary[col] = outliers
outlier_df = pd.DataFrame.from_dict(outlier_summary, orient="index", columns=["outlier_count"])
st.dataframe(outlier_df)

# --------------------------
# Trend detection
# --------------------------
st.subheader("📈 Trend Detection")
trend_result = detect_trends(cleaned)
trend_df = pd.DataFrame.from_dict(trend_result, orient="index", columns=["trend"]) if trend_result else pd.DataFrame()
st.dataframe(trend_df)

# --------------------------
# Anomaly detection
# --------------------------
st.subheader("📉 Anomaly Detection")
def detect_anomalies(df: pd.DataFrame, z_threshold: float = 3.0) -> dict:
    anomalies = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        if len(series) < 5 or series.nunique() < 5:
            continue
        z_scores = (series - series.mean()) / (series.std() + 1e-9)
        anomaly_count = (np.abs(z_scores) > z_threshold).sum()
        anomalies[col] = int(anomaly_count)
    return anomalies

anomaly_results = detect_anomalies(cleaned)
if anomaly_results:
    st.write("### Anomaly Summary")
    st.dataframe(pd.DataFrame.from_dict(anomaly_results, orient="index", columns=["anomaly_count"]))
    for col in anomaly_results.keys():
        series = cleaned[col]
        z_scores = (series - series.mean()) / (series.std() + 1e-9)
        anomaly_points = series[np.abs(z_scores) > 3]
        if not anomaly_points.empty:
            fig = px.line(series, y=series, title=f"Anomalies in {col}")
            fig.add_scatter(
                x=anomaly_points.index,
                y=anomaly_points.values,
                mode="markers",
                marker=dict(color="red", size=8),
                name="Anomaly"
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.success("No anomalies detected in numeric columns.")

# --------------------------
# Basic stats
# --------------------------
if st.checkbox("Show basic statistics (describe)"):
    st.write(cleaned.describe(include="all"))

# --------------------------
# Segment Analysis
# --------------------------
st.write("## Segment Analysis")
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
if len(categorical_cols) > 0:
    segment_col = st.selectbox("Select a categorical column for analysis:", categorical_cols)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) > 0:
        metric_col = st.selectbox("Select a numeric column to analyze:", numeric_cols)
        segment_summary = df.groupby(segment_col)[metric_col].sum().reset_index()
        segment_summary = segment_summary.sort_values(by=metric_col, ascending=False)
        st.write(f"### Top {segment_col} by {metric_col}")
        st.dataframe(segment_summary.head(10))
        fig = px.bar(segment_summary.head(10), x=segment_col, y=metric_col,
                     title=f"Top {segment_col} contributing to {metric_col}", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Ranking Insights
# --------------------------
example_insights = [
    {"insight": "Revenue ↑ in Mumbai", "impact": 0.8, "confidence": 0.9},
    {"insight": "Sales spike on Jan 15", "impact": 0.95, "confidence": 0.4},
    {"insight": "Product A ↓ in Q3", "impact": 0.6, "confidence": 0.7},
]
ranked = rank_insights(example_insights)
st.subheader("📊 Ranked Insights by Importance")
st.dataframe(ranked)
if not ranked.empty:
    st.bar_chart(ranked.set_index("insight")["score"])

# --------------------------
# Dynamic insights & narrative
# --------------------------
trends_summary = [{"metric": k, "trend": v} for k, v in trend_result.items()]
anomalies_summary = [{"metric": k, "count": v} for k, v in anomaly_results.items()]
insights = []
for metric, trend in trend_result.items():
    if trend in ["increasing", "decreasing"]:
        insights.append({
            "insight": f"{metric} is {trend}",
            "impact": np.random.uniform(0.5, 1.0),
            "confidence": np.random.uniform(0.6, 0.9)
        })
for metric, count in anomaly_results.items():
    if count > 0:
        insights.append({
            "insight": f"{count} anomalies detected in {metric}",
            "impact": np.random.uniform(0.6, 1.0),
            "confidence": np.random.uniform(0.5, 0.8)
        })
ranked = rank_insights(insights)

if not ranked.empty:
    insight_texts = [i["insight"] for i in ranked.to_dict(orient="records")]
    summary_output = summarize_insights(insight_texts)
else:
    insight_texts = []
    summary_output = ""

insights_output = {
    "trends": trend_result,
    "anomalies": anomaly_results,
    "segment_analysis": segment_summary.to_dict(orient="records") if 'segment_summary' in locals() else [],
    "ranked_insights": ranked.to_dict(orient="records") if not ranked.empty else [],
    "narrative_summary": summary_output
}

st.subheader("Insights (JSON)")
st.json(insights_output)
st.download_button(
    label="💾 Download Insights as JSON",
    data=json.dumps(insights_output, indent=2),
    file_name="insights_output.json",
    mime="application/json"
)

# --------------------------
# LLM summary
# --------------------------
if st.checkbox("💡 Ask LLM for summary"):
    prompt = f"Here are the insights: {insights_output}. Summarize in plain English."
    llm_summary = ask_openai(prompt)
    st.write("### 🤖 LLM Summary")
    st.write(llm_summary)

# --------------------------
# Narrative summary example
# --------------------------
insights_example = [
    "Sales increased by 20% in Q3 compared to Q2",
    "Customer churn dropped by 5% after loyalty program launch",
    "Mobile traffic now accounts for 60% of total visits"
]
summary_output = summarize_insights(insights_example)
st.subheader("📝 Narrative Insights Summary")
st.write(summary_output)

# --------------------------
# NLQ → SQL / Pandas
# --------------------------
st.subheader("💬 Ask in Natural Language")
nlq_input = st.text_input("enter your query (e.g., 'top 5 cities by sales'):")
test_queries = [
    "Show me top 5 cities by revenue",
    "Monthly revenue trend",
    "Which product category grew fastest?"
]
st.write("### 🔍 Quick Test Queries")
cols = st.columns(len(test_queries))
for i, tq in enumerate(test_queries):
    if cols[i].button(tq):
        nlq_input = tq

if nlq_input:
    engine_choice = st.radio("Select engine:", ["Pandas", "SQL"])
    st.info("Generating code...")
    code = nlq_to_code(nlq_input, engine=engine_choice.lower())
    st.code(code, language="python" if engine_choice =="Pandas" else "sql")
    
    result = None
    if st.button("Run Query"):
        result = execute_generated_code(code, engine=engine_choice.lower())
        st.subheader("📊 Query Result")
        if isinstance(result, pd.DataFrame):
            st.dataframe(result)
        else:
            st.write(result)

# --------------------------
# Download cleaned CSV
# --------------------------
@st.cache_data
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

csv_bytes = to_csv_bytes(cleaned)
st.download_button(
    "📥 Download cleaned CSV",
    data=csv_bytes,
    file_name=f"cleaned_{file_name.split('.')[0]}.csv",
    mime="text/csv"
)

# --------------------------
# Storage options
# --------------------------
st.subheader("🔎 Data Storage Options")
storage_option = st.radio(
    "Where do you want to store the dataset?",
    ["Pandas (in-memory)", "DuckDB (SQL engine)"]
)

if storage_option == "Pandas (in-memory)":
    st.success("✅ Data is stored in a Pandas DataFrame")
    st.write("Shape:", df.shape)
    st.dataframe(df.head(100))
elif storage_option == "DuckDB (SQL engine)":
    st.success("✅ Data loaded into DuckDB (query with SQL)")
    con = duckdb.connect(database=':memory:')
    con.register("uploaded_data", cleaned)
    st.markdown("### Example SQL Queries")
    query = st.text_area("Write your SQL query:", "SELECT * FROM uploaded_data LIMIT 10;")
    try:
        result = con.execute(query).df()
        st.dataframe(result)
    except Exception as e:
        st.error(f"SQL error: {e}")
