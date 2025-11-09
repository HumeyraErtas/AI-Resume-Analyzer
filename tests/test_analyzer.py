# tests/test_analyzer.py
from resume_analyzer import analyze_resume

def test_analyze_resume_basic():
    text = "Python developer with machine learning experience and good communication skills."
    result = analyze_resume(text)
    assert "overall_score" in result
    assert result["overall_score"] >= 0
    assert result["overall_score"] <= 100