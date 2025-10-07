from io import BytesIO
import re
import io
import os
import json
import duckdb
import pypandoc
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------
# ENV + OpenAI setup
# --------------------------
load_dotenv("password.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------
# Helper Functions
# --------------------------
def read_csv_with_sniff(file_bytes: bytes) -> pd.DataFrame:
    delimiters = [",", "\t", ";", "|"]
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"❌ Could not decode file: {e}")
        return pd.DataFrame()

    best_df, best_cols = None, 0
    for d in delimiters:
        try:
            df = pd.read_csv(io.StringIO(text), sep=d)
            if df.shape[1] > best_cols:
                best_cols, best_df = df.shape[1], df
        except Exception:
            continue

    if best_df is not None and not best_df.empty:
        return best_df
    else:
        st.error("❌ Could not detect valid CSV format. Try saving your file as UTF-8 CSV.")
        return pd.DataFrame()


def read_excel(file_bytes: bytes) -> dict:
    with io.BytesIO(file_bytes) as bio:
        xl = pd.ExcelFile(bio, engine="openpyxl")
        return {s: xl.parse(s) for s in xl.sheet_names}


def detect_trends(df):
    trends = {}
    for col in df.select_dtypes(include="number").columns:
        if df[col].dropna().empty:  # Skip columns that are entirely NaN or empty
            continue
        if df[col].is_monotonic_increasing:
            trends[col] = "increasing"
        elif df[col].is_monotonic_decreasing:
            trends[col] = "decreasing"
        else:
            trends[col] = "stable"
    return trends


def detect_anomalies(df, z_threshold=3.0):
    anomalies = {}
    for col in df.select_dtypes(include="number").columns:
        if df[col].dropna().empty:  # Skip empty columns
            continue
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        outliers = df[np.abs(z_scores) > z_threshold]
        if not outliers.empty:
            anomalies[col] = outliers[col]
    return anomalies



def rank_insights(insights: list):
    df = pd.DataFrame(insights)
    if df.empty:
        return pd.DataFrame(columns=["insight", "impact", "confidence", "score"])
    df["impact"] = pd.to_numeric(df.get("impact", 0), errors="coerce").fillna(0)
    df["confidence"] = pd.to_numeric(df.get("confidence", 0), errors="coerce").fillna(0)
    df["score"] = df["impact"] * df["confidence"]
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def summarize_insights(insights: list):
    if not insights:
        return "No insights available due to missing or invalid data."
    return "\n".join([f"• {i}" for i in insights])

def convert_for_json(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (pd.Series, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    else:
        return obj


def generate_markdown_report(insights_output):
    markdown = "# 📊 Data Insights Report\n\n"
    markdown += "## 🔍 Trends\n"
    for k, v in insights_output.get("trends", {}).items():
        markdown += f"- **{k}**: {v}\n"

    markdown += "\n## ⚠️ Anomalies\n"
    for k, v in insights_output.get("anomalies", {}).items():
        markdown += f"- **{k}**: {v}\n"

    markdown += "\n##🏆 Ranked Insights\n"
    for r in insights_output.get("ranked_insights", []):
        markdown += f"- **{r.get('insight', '')}** — Impact: {r.get('impact', '')}, Confidence: {r.get('confidence', '')}\n"

    return markdown


def markdown_to_pdf(markdown_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("AI Insights Report", styles["Title"]), Spacer(1, 12)]

    for line in markdown_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 6))
    doc.build(story)
    buffer.seek(0)
    return buffer

# --------------------------
# Safe Sql Check
# --------------------------
def is_safe_sql(query: str) ->bool:
    """
    Allow only SELECT Queries.
    Returns True if query is safe, False otherwise.
    """

    #Remove leading/trailing whitespaces and lowercase.
    q = query.strip().lower()

    #check it starts with 'select'
    if not q.startswith("select"):
        return False

    # prevent semiolons to stop multi-statement injections
    if ";" in q[:-1]:
        return False

    #prevent dangerous keyword
    unsafe_keywords =["insert", "update","delete","drop","alter","create", "truncate", "merge", "exec", "call"]
    for kw in unsafe_keywords:
        if kw in q:
            return False
    return True

# --------------------------
# Usage with pandas.read_sql / duckdb.query
# --------------------------
def run_query_safe (query: str, conn):
    if not is_safe_sql(query):
        raise ValueError("❌ Unsafe SQL query detected! Only SELECT statements allowed.")
    return conn.execute(query).fetchdf()   

# --------------------------
# MAIN FUNCTION
# --------------------------
def main():
    st.set_page_config(page_title="AI Data Insights App", layout="wide", page_icon="📊")
    st.sidebar.title("📁 Data Insights Dashboard")
    st.sidebar.markdown("---")

    uploaded_file = st.sidebar.file_uploader("📤 Upload CSV or Excel", type=["csv", "xlsx"])
    st.sidebar.markdown("---")

    if uploaded_file:
        file_bytes = uploaded_file.read()
        if uploaded_file.name.lower().endswith(".csv"):
            df = read_csv_with_sniff(file_bytes)
        else:
            sheets = read_excel(file_bytes)
            df = list(sheets.values())[0]

        #check for empty Dataframe immediately
        if df.empty:
            st.warning("⚠️ The uploaded file has no valid data. Please check the file.")
            st.stop()
        
        st.sidebar.success(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    else:
        st.sidebar.info("Upload a file to start.")
        st.stop()

    # --------------------------
    # Tabs
    # --------------------------
    tabs = st.tabs(["📤 Upload", "💡 Insights", "🤖 Ask Anything", "📈 Explore", "📦 Export"])

    # ========= UPLOAD TAB =========
    with tabs[0]:
        st.header("📤 Upload & Preview")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} cols")
        st.dataframe(df.head(10))

    # ========= INSIGHTS TAB =========
    
    with tabs[1]:
        st.header("💡 AI Insights")

        # ✅ Use cached functions for faster reruns
        @st.cache_data
        def detect_trends_cached(df):
            return detect_trends(df)

        @st.cache_data
        def detect_anomalies_cached(df):
            return detect_anomalies(df)

        trends = detect_trends_cached(df)
        anomalies = detect_anomalies_cached(df)

        # show warning if no trends or anomalies
        if not df.empty and not trends and not anomalies:
            st.write("⚠️ Data loaded but no trends or anomalies detected.")

        st.subheader("📈 Trends")
        if trends:
            trend_df = pd.DataFrame.from_dict(trends, orient="index", columns=["trend"])
            st.dataframe(trend_df)
        else:
            st.write("No clear trends detected.")

        st.subheader("📊 Trends + Anomalies")

        max_rows_plot = 1000  # ✅ limit rows for plotting

        for col in df.select_dtypes(include="number").columns:
            series = df[col].dropna()

            # ✅ Skip empty numeric columns
            if series.empty:
                continue

            # ✅ Limit number of rows plotted
            series_plot = series.head(max_rows_plot)

            fig = px.line(
                x=series_plot.index,
                y=series_plot.values,
                labels={"x": "Index", "y": col},
                title=f"Trend & Anomalies: {col} (showing first {max_rows_plot} rows)"
            )

            if col in anomalies:
                anomaly_points = anomalies[col]
                fig.add_scatter(
                    x=anomaly_points.index,
                    y=anomaly_points.values,
                    mode="markers",
                    marker=dict(color="red", size=10),
                    name="Anomaly"
                )
            st.plotly_chart(fig, use_container_width=True)

        # ✅ Build insights list safely
        insights_list = []
        for k, v in trends.items():
            if v in ["increasing", "decreasing"]:
                insights_list.append({
                    "insight": f"{k} is {v}",
                    "impact": np.random.rand(),
                    "confidence": np.random.rand()
                })

        for k, v in anomalies.items():
            if len(v) > 0:
                insights_list.append({
                    "insight": f"{len(v)} anomalies in {k}",
                    "impact": np.random.rand(),
                    "confidence": np.random.rand()
                })

        # ✅ Safe ranking only if insights exist
        ranked = rank_insights(insights_list) if insights_list else pd.DataFrame()
        if ranked.empty:
            st.info("⚠️ No valid insights could be generated from the data.")
        else:
            st.subheader("🏆 Ranked Insights")
            st.dataframe(ranked)

            fig_rank = px.bar(
                ranked,
                x="insight",
                y="score",
                color="score",
                text="score",
                title="Ranked Insights"
            )
            st.plotly_chart(fig_rank, use_container_width=True)

            summary = summarize_insights(ranked["insight"].tolist())

        st.text_area("📝 Narrative Summary", summary, height=150)

    # ========= ASK ANYTHING TAB =========
    with tabs[2]:
        st.header("🤖 Ask Anything (NLQ)")
        nlq = st.text_input("Ask a question about your data:")
        if st.button("Generate Code"):
            st.code("Function to generate code would run here.")  # placeholder

    # ========= EXPLORE TAB =========
    with tabs[3]:
        st.header("📈 Explore")
        st.dataframe(df)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if cat_cols and num_cols:
            cat = st.selectbox("Categorical Column", cat_cols)
            num = st.selectbox("Numeric Column", num_cols)
            pivot = df.groupby(cat)[num].sum().reset_index()
            st.dataframe(pivot)

            st.plotly_chart(px.bar(pivot, x=cat, y=num, color=num, text=num, title=f"{num} by {cat}"), use_container_width=True)
            st.plotly_chart(px.line(pivot, x=cat, y=num, markers=True, title=f"Trend of {num} by {cat}"), use_container_width=True)
            st.plotly_chart(px.scatter(pivot, x=cat, y=num, size=num, color=num, title=f"Scatter of {num} by {cat}"), use_container_width=True)

    # ========== EXPORT TAB =========
    with tabs[4]:
        st.header("📦 Export Data & Insights")

        # Export Cleaned Data
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Cleaned CSV",
            data=csv_bytes, 
            file_name="cleaned_data.csv", 
            mime="text/csv")

        # Export Insights JSON
        # Convert all objects to JSON serializable
        insights_output = {
            "trends": {k: convert_for_json(v) for k,v in trends.items()},
            "anomalies": {k: convert_for_json(v) for k,v in anomalies.items()},
            "ranked_insights": [ 
                {k: convert_for_json(v) for k,v in rec.items()}
                for rec in ranked.to_dict(orient="records")
                ] if not ranked.empty else [],
            "summary": summary,
        }

        st.download_button(
            "💾 Download Insights JSON",
            data=json.dumps(insights_output, indent=2),
            file_name="insights.json",
            mime="application/json",
        )

        ## --- Export Markdown Report ---
        st.subheader("📤 Export Report")

        if "insights_output" in locals() or "insights_output" in globals():
            markdown_report = generate_markdown_report(insights_output)
        
            st.download_button(
                "📄 Download Markdown Report",
                data=markdown_report,
                file_name="insights_report.md",
                mime="text/markdown",
            )

            pdf_buffer = markdown_to_pdf(markdown_report)
            st.download_button(
                "🧾 Download PDF Report",
                data=pdf_buffer.getvalue(),  # ✅ Added .getvalue()
                file_name="insights_report.pdf",
                mime="application/pdf",
            )
        else:
            st.info("insights available to export yet. Please run the analysis first.")

if __name__ == "__main__":
    main()
