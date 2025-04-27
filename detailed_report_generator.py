# detailed_report_generator.py
import logging
from fpdf import FPDF, XPos, YPos
import tempfile
import os
import io
import base64
from datetime import datetime
import requests

# Configure logging
logger = logging.getLogger(__name__)

class ReportGenerator:
    """Class for generating detailed PDF reports for YouTube analysis"""
    
    def __init__(self):
        """Initialize the report generator"""
        logger.info("Initializing Report Generator")
        # Don't automatically download fonts - we'll handle fonts differently
    
    def format_value(self, metric_type, value):
        """Format numeric values for display in reports"""
        try:
            if metric_type in ['views', 'likes', 'comments']:
                return f"{int(value):,}"
            elif metric_type == 'engagement_rate':
                return f"{float(value):.2f}%"
            else:
                return str(value)
        except (ValueError, TypeError):
            return "N/A"
    
    def create_pdf_report(self, analysis_results, analysis_type, channel_info=None):
        """Create a PDF report for the analysis results"""
        try:
            # Check if analysis_results is a dictionary
            if not isinstance(analysis_results, dict):
                logger.error(f"Invalid analysis_results type: {type(analysis_results)}")
                return None
            
            # Create PDF object with built-in fonts only
            pdf = FPDF()
            
            # Use built-in fonts that support basic characters
            # Arial has better Unicode support than Helvetica
            font_family = 'Arial'
            
            pdf.set_margins(20, 20, 20)  # Increase margins (left, top, right)
            pdf.add_page()
            
            # Set default font and page settings
            pdf.set_font(font_family, size=10)
            pdf.set_auto_page_break(auto=True, margin=20)
            
            # Title
            pdf.set_font(font_family, 'B', 14)
            if channel_info:
                channel_title = channel_info.get('title', 'Unknown Channel')
                title = f'YouTube Analysis Report - {channel_title}'
            else:
                title = 'YouTube Analysis Report'
            pdf.cell(w=0, h=10, text=title, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            # Add timestamp
            pdf.set_font(font_family, size=10)
            pdf.cell(w=0, h=8, text=f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                    new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(4)
            
            # Basic metrics
            if 'total_metrics' in analysis_results:
                pdf.set_font(font_family, 'B', 12)
                pdf.cell(w=0, h=8, text='Performance Metrics', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font(font_family, size=10)
                
                metrics = analysis_results['total_metrics']
                for metric, value in metrics.items():
                    # Replace currency symbols with their names to avoid encoding issues
                    metric_name = str(metric).replace('₹', 'INR')
                    text = f'{metric_name.title()}: {self.format_value(metric, value)}'
                    pdf.cell(w=0, h=6, text=text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(4)
            
            # Averages
            if 'averages' in analysis_results:
                pdf.set_font(font_family, 'B', 12)
                pdf.cell(w=0, h=8, text='Average Performance', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font(font_family, size=10)
                
                avgs = analysis_results['averages']
                for metric, value in avgs.items():
                    # Replace currency symbols with their names
                    metric_name = str(metric).replace('₹', 'INR')
                    text = f'Average {metric_name.title()}: {self.format_value(metric, value)}'
                    pdf.cell(w=0, h=6, text=text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(4)
            
            # Sentiment Analysis
            if 'sentiment_analysis' in analysis_results:
                pdf.set_font(font_family, 'B', 12)
                pdf.cell(w=0, h=8, text='Sentiment Analysis', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font(font_family, size=10)
                
                sentiment_data = analysis_results['sentiment_analysis']
                # Skip non-numeric values like 'gemini_enhanced_analysis'
                numeric_sentiments = {k: v for k, v in sentiment_data.items() if isinstance(v, (int, float))}
                
                total = sum(numeric_sentiments.values())
                if total > 0:
                    for sentiment, count in numeric_sentiments.items():
                        percentage = (count / total) * 100
                        text = f'{sentiment}: {percentage:.1f}%'
                        pdf.cell(w=0, h=6, text=text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(4)
            
            # Gemini Analysis
            if 'gemini_analysis' in analysis_results:
                pdf.add_page()  # Start on new page for detailed analysis
                pdf.set_font(font_family, 'B', 12)
                pdf.cell(w=0, h=8, text='AI-Powered Analysis', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font(font_family, size=10)
                
                # Process analysis text
                analysis_text = analysis_results['gemini_analysis']
                
                # Split into paragraphs and clean
                paragraphs = [p.strip() for p in str(analysis_text).split('\n') if p.strip()]
                
                for paragraph in paragraphs:
                    # Handle bullet points
                    if paragraph.startswith('•') or paragraph.startswith('-'):
                        pdf.cell(w=10, h=6, text='•', new_x=XPos.RIGHT)
                        pdf.multi_cell(w=0, h=6, text=paragraph[1:].strip())
                    else:
                        # Regular paragraph
                        pdf.multi_cell(w=0, h=6, text=paragraph)
                    pdf.ln(2)
            
            # Enhanced Analysis
            if 'enhanced_analysis' in analysis_results:
                pdf.add_page()
                pdf.set_font(font_family, 'B', 12)
                pdf.cell(w=0, h=8, text='Enhanced Sentiment Analysis', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font(font_family, size=10)
                
                enhanced = analysis_results['enhanced_analysis']
                
                # Add detailed interpretation
                if "detailed_interpretation" in enhanced and enhanced["detailed_interpretation"]:
                    pdf.set_font(font_family, 'B', 11)
                    pdf.cell(w=0, h=8, text='Detailed Interpretation', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.set_font(font_family, size=10)
                    pdf.multi_cell(w=0, h=6, text=enhanced["detailed_interpretation"])
                    pdf.ln(4)
                
                # Add deeper insights
                if "deeper_insights" in enhanced and enhanced["deeper_insights"]:
                    pdf.set_font(font_family, 'B', 11)
                    pdf.cell(w=0, h=8, text='Deeper Insights', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.set_font(font_family, size=10)
                    
                    for insight in enhanced["deeper_insights"]:
                        pdf.cell(w=10, h=6, text='•', new_x=XPos.RIGHT)
                        pdf.multi_cell(w=0, h=6, text=insight)
                    pdf.ln(4)
                
                # Add actionable recommendations
                if "actionable_recommendations" in enhanced and enhanced["actionable_recommendations"]:
                    pdf.set_font(font_family, 'B', 11)
                    pdf.cell(w=0, h=8, text='Actionable Recommendations', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.set_font(font_family, size=10)
                    
                    for rec in enhanced["actionable_recommendations"]:
                        pdf.cell(w=10, h=6, text='•', new_x=XPos.RIGHT)
                        pdf.multi_cell(w=0, h=6, text=rec)
                    pdf.ln(4)
                
                # Add content strategy suggestions
                if "content_strategy_suggestions" in enhanced and enhanced["content_strategy_suggestions"]:
                    pdf.set_font(font_family, 'B', 11)
                    pdf.cell(w=0, h=8, text='Content Strategy Suggestions', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.set_font(font_family, size=10)
                    
                    for suggestion in enhanced["content_strategy_suggestions"]:
                        pdf.cell(w=10, h=6, text='•', new_x=XPos.RIGHT)
                        pdf.multi_cell(w=0, h=6, text=suggestion)
                    pdf.ln(4)
                
                # Add identified themes
                if "identified_themes" in enhanced and enhanced["identified_themes"]:
                    pdf.set_font(font_family, 'B', 11)
                    pdf.cell(w=0, h=8, text='Identified Themes', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.set_font(font_family, size=10)
                    
                    for theme in enhanced["identified_themes"]:
                        pdf.cell(w=10, h=6, text='•', new_x=XPos.RIGHT)
                        pdf.multi_cell(w=0, h=6, text=theme)
            
            # Get the PDF as bytes directly
            pdf_bytes = pdf.output()
            
            # Return the bytes directly without additional encoding
            return pdf_bytes
            
        except Exception as e:
            logger.error(f"Error creating PDF report: {e}")
            return None

# Create a singleton instance
report_generator = ReportGenerator()