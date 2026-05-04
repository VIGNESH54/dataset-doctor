import pytest
from unittest.mock import patch, MagicMock
from dataset_doctor.llm import LLMInsightGenerator

def test_llm_insight_generator_no_api_key():
    llm = LLMInsightGenerator(api_key="")
    insights = llm.generate_insights({})
    assert insights["overall_assessment"] == "Could not generate insights."
    assert "LLM insights unavailable." in insights["critical_issues"]

@patch('dataset_doctor.llm.Anthropic')
def test_llm_insight_generator_success(mock_anthropic):
    # Mock the API response
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    
    mock_message = MagicMock()
    mock_content = MagicMock()
    mock_content.text = '```json\n{"critical_issues": ["i1"], "cleaning_steps": ["s1"], "warnings": ["w1"], "overall_assessment": "Good"}\n```'
    mock_message.content = [mock_content]
    
    mock_client.messages.create.return_value = mock_message
    
    llm = LLMInsightGenerator(api_key="fake_key")
    insights = llm.generate_insights({"test": "data"})
    
    assert insights["overall_assessment"] == "Good"
    assert insights["critical_issues"] == ["i1"]
    assert insights["cleaning_steps"] == ["s1"]
    assert insights["warnings"] == ["w1"]
