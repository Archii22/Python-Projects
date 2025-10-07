import pytest
import pandas as pd
import numpy as np
from streamlit_app import detect_trends, detect_anomalies, rank_insights, summarize_insights

# ----------------------------
# SAMPLE DATA FOR TESTING
# ----------------------------
@pytest.fixture
def sample_df():
    data = {
        "Units_sold": [10, 20, 30, 40, 50],
        "Revenue": [100, 200, 300, 400, 500],
        "Profit": [20, 40, 60, 80, 100],
        "Stable_column": [5, 5, 5, 5, 5],
    }
    return pd.DataFrame(data)

# ----------------------------
# TEST detect_trends
# ----------------------------
def test_detect_trends_returns_dict(sample_df):
    trends = detect_trends(sample_df)
    assert isinstance(trends, dict)
    assert "Units_sold" in trends
    assert "Revenue" in trends
    assert "Profit" in trends
    # Stable column with zero variance should be skipped
    assert "Stable_column" not in trends

def test_detect_trends_values(sample_df):
    trends = detect_trends(sample_df)
    for v in trends.values():
        assert v in ["increasing", "decreasing", "stable"]

# ----------------------------
# TEST detect_anomalies
# ----------------------------
def test_detect_anomalies_no_anomalies(sample_df):
    anomalies = detect_anomalies(sample_df, z_threshold=10.0)  # very high threshold
    assert anomalies == {}

def test_detect_anomalies_with_anomalies():
    df = pd.DataFrame({"A": [1, 2, 3, 100, 5, 6, 7]})
    anomalies = detect_anomalies(df, z_threshold=2.0)
    assert "A" in anomalies
    assert 100 in anomalies["A"].values

# ----------------------------
# TEST rank_insights
# ----------------------------
def test_rank_insights_basic():
    insights = [
        {"insight": "A is increasing", "impact": 0.5, "confidence": 0.8},
        {"insight": "B has anomalies", "impact": 0.9, "confidence": 0.5}
    ]
    ranked = rank_insights(insights)
    assert not ranked.empty
    assert "score" in ranked.columns
    # highest score should come first
    assert ranked.iloc[0]["score"] >= ranked.iloc[1]["score"]

def test_rank_insights_empty():
    ranked = rank_insights([])
    assert ranked.empty
    assert list(ranked.columns) == ["insight", "impact", "confidence", "score"]

# ----------------------------
# TEST summarize_insights
# ----------------------------
def test_summarize_insights_basic():
    insights = ["Trend A increasing", "Trend B decreasing"]
    summary = summarize_insights(insights)
    assert "• Trend A increasing" in summary
    assert "• Trend B decreasing" in summary

def test_summarize_insights_empty():
    summary = summarize_insights([])
    assert summary == "No insights available."

# ----------------------------
# TEST NLQ (OpenAI) MOCK
# ----------------------------
def test_nlq_placeholder(monkeypatch):
    from streamlit_app import client

    # Fake response class
    class FakeResponse:
        def __init__(self):
            self.choices = [{"message": {"content": "print('Hello world')"}}]

    # Monkeypatch the OpenAI call
    def fake_create(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(client.chat.completions, "create", fake_create)

    # Simulate NLQ question
    question = "show me python code to sum a column in pandas"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}]
    )

    content = response.choices[0]["message"]["content"]
    assert "print" in content
    assert "Hello world" in content
