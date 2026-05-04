"""HTML Report generator for Data Quality Analyzer."""

import base64
from io import BytesIO
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
from typing import Dict, Any

class ReportGenerator:
    """Generates an HTML report from dataset profile and scores."""

    def __init__(self, template_dir: str = None):
        """Initialize the report generator."""
        if template_dir is None:
            # Default to the templates directory in the package
            template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )

    def _generate_gauge_chart(self, score: float) -> str:
        """Generate a gauge chart for the quality score as a base64 string."""
        fig, ax = plt.subplots(figsize=(4, 3))
        
        # Determine color based on score
        if score >= 80:
            color = 'green'
        elif score >= 60:
            color = 'orange'
        else:
            color = 'red'

        # Create a simple pie chart that looks like a gauge
        ax.pie([score, 100-score], colors=[color, '#e0e0e0'], startangle=90, counterclock=False)
        ax.add_artist(plt.Circle((0,0), 0.70, fc='white'))
        
        plt.text(0, 0, f"{score:.1f}", ha='center', va='center', fontsize=20, fontweight='bold')
        
        plt.axis('equal')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def generate_report(self, 
                       profile: Dict[str, Any], 
                       score_data: Dict[str, Any], 
                       llm_insights: Dict[str, Any],
                       output_path: str) -> None:
        """Render the HTML report and save to file."""
        template = self.env.get_template("report.html")
        
        gauge_b64 = self._generate_gauge_chart(score_data.get("score", 0))
        
        html_content = template.render(
            profile=profile,
            score_data=score_data,
            llm_insights=llm_insights,
            gauge_chart=gauge_b64
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
