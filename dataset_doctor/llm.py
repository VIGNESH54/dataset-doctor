"""LLM Insights Generator using Anthropic Claude."""

import json
import os
import logging
from typing import Dict, Any, Optional
try:
    from anthropic import Anthropic, APIError
except ImportError:
    pass

logger = logging.getLogger(__name__)

class LLMInsightGenerator:
    """Generates insights and recommendations using Claude LLM."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Anthropic API key."""
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if key:
            self.client = Anthropic(api_key=key)
        else:
            self.client = None

    def generate_insights(self, profile_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from dataset profile using Claude Haiku."""
        if not self.client:
            logger.warning("No Anthropic API key found. Skipping LLM insights.")
            return self._empty_insights()

        prompt = self._build_prompt(profile_summary)
        
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                temperature=0.2,
                system="You are an expert Data Scientist analyzing data quality. Respond ONLY with valid JSON matching the requested schema.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the JSON response
            result_text = response.content[0].text.strip()
            # Handle potential markdown code block formatting
            if result_text.startswith("```json"):
                result_text = result_text[7:-3].strip()
            elif result_text.startswith("```"):
                result_text = result_text[3:-3].strip()
                
            return json.loads(result_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return self._empty_insights()
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return self._empty_insights()

    def _build_prompt(self, profile_summary: Dict[str, Any]) -> str:
        """Construct the prompt for the LLM."""
        # Truncate summary if it's too large, but for now we convert to string
        summary_str = json.dumps(profile_summary, default=str)[:10000] # Limit size

        return f"""
Analyze the following dataset profile summary and provide data quality insights.

DATASET PROFILE:
{summary_str}

Return a JSON object with EXACTLY the following structure:
{{
    "critical_issues": ["issue 1", "issue 2", "issue 3"],
    "cleaning_steps": ["python code step 1", "python code step 2"],
    "warnings": ["warning about potential target leakage or ID columns"],
    "overall_assessment": "A brief 2-sentence summary of the dataset quality"
}}

Focus on:
1. Missing values and outliers.
2. Suspicious columns (e.g., target leakage, completely unique ID columns mistakenly included in ML).
3. Type inconsistencies.
Ensure the output is pure JSON.
"""

    def _empty_insights(self) -> Dict[str, Any]:
        """Return an empty insights dictionary on failure or missing API key."""
        return {
            "critical_issues": ["LLM insights unavailable."],
            "cleaning_steps": [],
            "warnings": [],
            "overall_assessment": "Could not generate insights."
        }
