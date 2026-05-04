import os
import tempfile
from dataset_doctor.report import ReportGenerator

def test_report_generator_html_output():
    report_gen = ReportGenerator()
    
    profile = {
        "num_rows": 100,
        "num_cols": 2,
        "columns": {
            "A": {"missing_percentage": 10.0, "missing_count": 10, "unique_count": 50, "inferred_type": "int", "is_numeric": True, "mean": 5.0, "outlier_count": 2},
            "B": {"missing_percentage": 0.0, "missing_count": 0, "unique_count": 100, "inferred_type": "object", "is_numeric": False}
        }
    }
    
    score_data = {
        "score": 85.5,
        "warnings": ["Warning 1"]
    }
    
    llm_insights = {
        "overall_assessment": "Looks okay.",
        "critical_issues": ["Issue 1"],
        "cleaning_steps": ["step = 1"],
        "warnings": []
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.html")
        report_gen.generate_report(profile, score_data, llm_insights, output_path)
        
        assert os.path.exists(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        assert "Data Quality Analyzer Report" in content
        assert "85.5/100" in content
        assert "Warning 1" in content
        assert "Looks okay." in content
        assert "Issue 1" in content
