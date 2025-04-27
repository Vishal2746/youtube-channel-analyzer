import logging
from transformers import pipeline
import pandas as pd
from typing import List, Dict, Any
import numpy as np
from datetime import datetime
import google.generativeai as genai
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YouTubeAnalysisModel:
    def __init__(self):
        """Initialize the sentiment analysis model and Gemini API."""
        try:
            # Initialize sentiment analysis pipeline with Twitter RoBERTa model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            logger.info("Twitter RoBERTa sentiment analysis model initialized successfully")
            
            # Initialize Gemini API if key is available
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            self.gemini_available = False
            
            if gemini_api_key:
                try:
                    genai.configure(api_key=gemini_api_key)
                    
                    # Configure safety settings to be more permissive for content analysis
                    safety_settings = [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        }
                    ]
                    
                    # Initialize the model with safety settings
                    self.gemini_model = genai.GenerativeModel(
                        model_name='gemini-2.0-flash',
                        safety_settings=safety_settings
                    )
                    self.gemini_available = True
                    logger.info("Gemini API initialized successfully")
                except Exception as e:
                    logger.error(f"Error initializing Gemini API: {str(e)}")
                    self.gemini_available = False
            else:
                logger.warning("Gemini API key not found. Enhanced analysis will not be available.")
                
        except Exception as e:
            logger.error(f"Error initializing sentiment analysis model: {str(e)}")
            raise

    def analyze_comments(self, comments: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment of YouTube comments.
        
        Args:
            comments (List[str]): List of comment texts to analyze
            
        Returns:
            Dict[str, Any]: Analysis results including sentiment distribution
                           and key insights
        """
        try:
            if not comments:
                return {
                    "error": "No comments provided for analysis",
                    "sentiment_distribution": {},
                    "insights": []
                }

            # Analyze sentiment for each comment
            results = []
            for comment in comments:
                try:
                    sentiment_scores = self.sentiment_analyzer(comment)[0]
                    # Map Twitter model labels to standard labels
                    label_mapping = {
                        'LABEL_0': 'NEGATIVE',
                        'LABEL_1': 'NEUTRAL',
                        'LABEL_2': 'POSITIVE'
                    }
                    results.append({
                        "text": comment,
                        "sentiment": label_mapping[max(sentiment_scores, key=lambda x: x['score'])['label']],
                        "scores": {label_mapping[score['label']]: score['score'] for score in sentiment_scores}
                    })
                except Exception as e:
                    logger.warning(f"Error analyzing comment: {str(e)}")
                    continue

            # Calculate sentiment distribution
            sentiment_counts = pd.Series([r['sentiment'] for r in results]).value_counts()
            sentiment_distribution = {
                sentiment: count / len(results)
                for sentiment, count in sentiment_counts.items()
            }

            # Generate insights
            insights = self._generate_sentiment_insights(results)
            
            # Create base analysis results
            analysis_results = {
                "sentiment_distribution": sentiment_distribution,
                "insights": insights,
                "sample_comments": results[:5]  # Include first 5 analyzed comments
            }
            
            # Enhance analysis with Gemini if available
            if self.gemini_available:
                enhanced_analysis = self._enhance_with_gemini(results, sentiment_distribution, insights)
                if enhanced_analysis:
                    analysis_results["enhanced_analysis"] = enhanced_analysis
            
            return analysis_results

        except Exception as e:
            logger.error(f"Error in comment analysis: {str(e)}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "sentiment_distribution": {},
                "insights": []
            }

    def _generate_sentiment_insights(self, results: List[Dict]) -> List[str]:
        """
        Generate insights from sentiment analysis results.
        
        Args:
            results (List[Dict]): List of analyzed comments with sentiment scores
            
        Returns:
            List[str]: List of insights about comment sentiment
        """
        insights = []
        
        try:
            # Calculate average sentiment scores
            avg_scores = {
                label: np.mean([r['scores'][label] for r in results])
                for label in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
            }
            
            # Generate insights based on sentiment distribution
            if avg_scores.get('POSITIVE', 0) > 0.7:
                insights.append("Comments are overwhelmingly positive")
            elif avg_scores.get('NEGATIVE', 0) > 0.7:
                insights.append("Comments show strong negative sentiment")
            else:
                insights.append("Comments show mixed sentiment")
                
            # Add more specific insights based on score thresholds
            if avg_scores.get('POSITIVE', 0) > 0.6:
                insights.append("Content is well-received by the audience")
            if avg_scores.get('NEGATIVE', 0) > 0.4:
                insights.append("Some content may need improvement based on negative feedback")
                
            return insights
            
        except Exception as e:
            logger.error(f"Error generating sentiment insights: {str(e)}")
            return ["Unable to generate detailed insights"]

    def _enhance_with_gemini(self, results: List[Dict], sentiment_distribution: Dict, insights: List[str]) -> Dict[str, Any]:
        """
        Enhance sentiment analysis with Gemini API for better understanding.
        
        Args:
            results (List[Dict]): List of analyzed comments with sentiment scores
            sentiment_distribution (Dict): Distribution of sentiments
            insights (List[str]): Basic insights generated from sentiment analysis
            
        Returns:
            Dict[str, Any]: Enhanced analysis from Gemini
        """
        if not self.gemini_available:
            logger.warning("Gemini enhancement requested but API not available")
            return None
            
        try:
            # Prepare data for Gemini
            sample_comments = results[:10]  # Use first 10 comments as samples
            
            # Format sentiment distribution for prompt
            sentiment_dist_str = "\n".join([
                f"- {sentiment}: {percentage:.2%}"
                for sentiment, percentage in sentiment_distribution.items()
            ])
            
            # Format sample comments for prompt
            sample_comments_str = "\n".join([
                f"- \"{comment['text'][:100]}{'...' if len(comment['text']) > 100 else ''}\" (Sentiment: {comment['sentiment']})"
                for comment in sample_comments
            ])
            
            # Format basic insights for prompt
            insights_str = "\n".join([f"- {insight}" for insight in insights])
            
            # Create prompt for Gemini
            prompt = f"""As a YouTube sentiment analysis expert, analyze and enhance the following sentiment analysis results to provide deeper insights:

SENTIMENT DISTRIBUTION:
{sentiment_dist_str}

SAMPLE COMMENTS:
{sample_comments_str}

BASIC INSIGHTS:
{insights_str}

Please provide:
1. A detailed interpretation of the sentiment patterns
2. Deeper insights into audience reactions and emotions
3. Actionable recommendations based on the sentiment analysis
4. Potential content strategy adjustments based on audience sentiment
5. Key themes or topics identified in the comments

Format your response as a structured JSON with the following keys:
- detailed_interpretation (string)
- deeper_insights (list of strings)
- actionable_recommendations (list of strings)
- content_strategy_suggestions (list of strings)
- identified_themes (list of strings)
"""

            # Call Gemini API
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.2,  # Lower temperature for more focused response
                    'response_mime_type': 'application/json',
                }
            )
            
            # Process response
            if response.text:
                try:
                    # Try to parse as JSON
                    enhanced_analysis = json.loads(response.text)
                    logger.info("Successfully enhanced sentiment analysis with Gemini")
                    return enhanced_analysis
                except json.JSONDecodeError:
                    # If not valid JSON, extract structured content manually
                    logger.warning("Gemini response not in valid JSON format, extracting manually")
                    
                    # Create a structured response manually
                    text = response.text
                    sections = {
                        "detailed_interpretation": "",
                        "deeper_insights": [],
                        "actionable_recommendations": [],
                        "content_strategy_suggestions": [],
                        "identified_themes": []
                    }
                    
                    # Simple parsing of the response text
                    current_section = None
                    for line in text.split('\n'):
                        line = line.strip()
                        if "detailed_interpretation" in line.lower():
                            current_section = "detailed_interpretation"
                        elif "deeper_insights" in line.lower():
                            current_section = "deeper_insights"
                        elif "actionable_recommendations" in line.lower():
                            current_section = "actionable_recommendations"
                        elif "content_strategy" in line.lower():
                            current_section = "content_strategy_suggestions"
                        elif "identified_themes" in line.lower():
                            current_section = "identified_themes"
                        elif line and current_section:
                            if current_section == "detailed_interpretation":
                                sections[current_section] += line + " "
                            elif line.startswith("- ") or line.startswith("* "):
                                sections[current_section].append(line[2:])
                    
                    return sections
            
            logger.warning("Gemini API returned empty response")
            return None
            
        except Exception as e:
            logger.error(f"Error enhancing analysis with Gemini: {str(e)}")
            return None

    def format_analysis_for_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Format sentiment analysis results for reporting.
        
        Args:
            analysis_results (Dict[str, Any]): Results from analyze_comments
            
        Returns:
            str: Formatted report string
        """
        try:
            if "error" in analysis_results:
                return f"Analysis Error: {analysis_results['error']}"

            report = ["📊 Sentiment Analysis Results\n"]
            
            # Add sentiment distribution
            report.append("\nSentiment Distribution:")
            for sentiment, percentage in analysis_results["sentiment_distribution"].items():
                report.append(f"- {sentiment}: {percentage:.1%}")
            
            # Add enhanced analysis if available
            if "enhanced_analysis" in analysis_results:
                enhanced = analysis_results["enhanced_analysis"]
                
                # Add detailed interpretation
                if "detailed_interpretation" in enhanced and enhanced["detailed_interpretation"]:
                    report.append("\n🔍 Detailed Interpretation:")
                    report.append(enhanced["detailed_interpretation"])
                
                # Add deeper insights
                if "deeper_insights" in enhanced and enhanced["deeper_insights"]:
                    report.append("\n💡 Deeper Insights:")
                    for insight in enhanced["deeper_insights"]:
                        report.append(f"- {insight}")
                
                # Add actionable recommendations
                if "actionable_recommendations" in enhanced and enhanced["actionable_recommendations"]:
                    report.append("\n🎯 Actionable Recommendations:")
                    for rec in enhanced["actionable_recommendations"]:
                        report.append(f"- {rec}")
                
                # Add content strategy suggestions
                if "content_strategy_suggestions" in enhanced and enhanced["content_strategy_suggestions"]:
                    report.append("\n📝 Content Strategy Suggestions:")
                    for suggestion in enhanced["content_strategy_suggestions"]:
                        report.append(f"- {suggestion}")
                
                # Add identified themes
                if "identified_themes" in enhanced and enhanced["identified_themes"]:
                    report.append("\n🔑 Identified Themes:")
                    for theme in enhanced["identified_themes"]:
                        report.append(f"- {theme}")
            else:
                # Add basic insights if no enhanced analysis
                if analysis_results["insights"]:
                    report.append("\nKey Insights:")
                    for insight in analysis_results["insights"]:
                        report.append(f"- {insight}")
            
            # Add sample comments
            if "sample_comments" in analysis_results:
                report.append("\nSample Comments:")
                for comment in analysis_results["sample_comments"]:
                    report.append(f"- {comment['text'][:100]}... ({comment['sentiment']})")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error formatting analysis report: {str(e)}")
            return "Error formatting analysis results"

# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = YouTubeAnalysisModel()
    
    # Example comments
    example_comments = [
        "This video was amazing! I learned so much from it.",
        "I didn't really like this content, it was boring.",
        "Great explanation of the topic, very clear and helpful.",
        "The audio quality could be better, but the content is good.",
        "I've been waiting for this video for so long! Worth the wait!"
    ]
    
    # Analyze comments
    analysis = model.analyze_comments(example_comments)
    
    # Format and print report
    report = model.format_analysis_for_report(analysis)
    print(report)