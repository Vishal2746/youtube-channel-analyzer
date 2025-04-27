import streamlit as st
import os
import logging
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from time import sleep
from itertools import islice
from fpdf import FPDF
import matplotlib.pyplot as plt
import plotly.io as pio
import io
import base64
import tempfile
import re
import time
import numpy as np
from collections import defaultdict
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Import necessary functions from youtube_analyzer
from youtube_analyzer import (
    get_channel_id,
    filter_videos_by_date_grouped,
    analyze_channel_performance,
    APIQuotaExceeded,
    get_youtube_videos_grouped,
    sentiment_pipeline
)

# --- Basic Configuration ---
st.set_page_config(page_title="YouTube Channel Analyzer", page_icon="📊", layout="wide")
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- API Key Loading ---
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
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
    model = genai.GenerativeModel(model_name='gemini-2.0-flash',
                                safety_settings=safety_settings)
    logger.info("Gemini Flash model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Gemini model: {e}")
    model = None

# --- Enhanced Channel Comparison Class ---
class EnhancedChannelComparison:
    """Enhanced comparison system for YouTube channels that focuses on direct comparisons"""
    
    def __init__(self):
        """Initialize the comparison system"""
        logger.info("Initializing Enhanced Channel Comparison system...")
        
    def compare_channels(self, all_channel_results, channel_info):
        """Compare multiple channels directly against each other"""
        if not all_channel_results or len(all_channel_results) < 2:
            logger.warning("Need at least 2 channels to compare")
            return pd.DataFrame(), pd.DataFrame()
            
        logger.info(f"Comparing {len(all_channel_results)} channels...")
        
        # Create comparison dataframe
        comparison_data = []
        
        for channel_id, results in all_channel_results.items():
            channel_name = channel_info.get(channel_id, {}).get('title', channel_id)
            channel_url = channel_info.get(channel_id, {}).get('url', '')
            
            # Extract metrics
            total_metrics = results.get('total_metrics', {})
            averages = results.get('averages', {})
            sentiment = results.get('sentiment_analysis', {})
            
            # Calculate engagement rate
            total_views = total_metrics.get('views', 0)
            total_interactions = total_metrics.get('likes', 0) + total_metrics.get('comments', 0)
            engagement_rate = (total_interactions / total_views * 100) if total_views > 0 else 0
            
            # Create entry
            entry = {
                'Channel ID': channel_id,
                'Channel Name': channel_name,
                'Channel URL': channel_url,
                'Videos Analyzed': results.get('num_videos', 0),
                'Total Views': total_metrics.get('views', 0),
                'Total Likes': total_metrics.get('likes', 0),
                'Total Comments': total_metrics.get('comments', 0),
                'Avg Views': averages.get('views', 0),
                'Avg Likes': averages.get('likes', 0),
                'Avg Comments': averages.get('comments', 0),
                'Engagement Rate': engagement_rate,
                'Positive Sentiment': sentiment.get('POSITIVE', 0),
                'Neutral Sentiment': sentiment.get('NEUTRAL', 0),
                'Negative Sentiment': sentiment.get('NEGATIVE', 0)
            }
            
            comparison_data.append(entry)
        
        # Create raw dataframe
        raw_df = pd.DataFrame(comparison_data)
        
        # Create formatted dataframe for display
        formatted_df = raw_df.copy()
        
        # Format numeric columns
        numeric_cols = ['Total Views', 'Total Likes', 'Total Comments', 
                       'Avg Views', 'Avg Likes', 'Avg Comments']
        
        for col in numeric_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{int(x):,}")
        
        # Format percentage columns
        pct_cols = ['Engagement Rate']
        for col in pct_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}%")
        
        logger.info(f"Created comparison dataframes with {len(comparison_data)} channels")
        return formatted_df, raw_df
    
    def generate_comparison_visualizations(self, raw_df):
        """Generate visualizations comparing channels directly against each other"""
        try:
            if raw_df.empty:
                logger.warning("No data provided for comparison visualizations")
                return {}
            
            visualizations = {}
            
            # 1. Total Views Comparison
            if 'Total Views' in raw_df.columns and 'Channel Name' in raw_df.columns:
                fig_views = px.bar(
                    raw_df, 
                    x='Channel Name', 
                    y='Total Views',
                    title='Total Views Comparison',
                    color='Total Views',
                    color_continuous_scale='Viridis',
                    text_auto='.2s'
                )
                fig_views.update_traces(textposition='outside')
                visualizations['total_views'] = fig_views
            
            # 2. Engagement Rate Comparison
            if 'Engagement Rate' in raw_df.columns and 'Channel Name' in raw_df.columns:
                fig_engagement = px.bar(
                    raw_df, 
                    x='Channel Name', 
                    y='Engagement Rate',
                    title='Engagement Rate Comparison',
                    color='Engagement Rate',
                    color_continuous_scale='Viridis',
                    text_auto='.2f'
                )
                fig_engagement.update_traces(textposition='outside')
                visualizations['engagement_rate'] = fig_engagement
            
            # 3. Average Metrics Comparison (Radar Chart)
            if all(col in raw_df.columns for col in ['Channel Name', 'Avg Views', 'Avg Likes', 'Avg Comments']):
                # Normalize metrics for radar chart
                radar_df = raw_df.copy()
                metrics = ['Avg Views', 'Avg Likes', 'Avg Comments', 'Engagement Rate']
                
                # Create radar chart for each channel
                fig_radar = go.Figure()
                
                for _, row in radar_df.iterrows():
                    channel_name = row['Channel Name']
                    
                    # Normalize values for better visualization
                    values = []
                    for metric in metrics:
                        if metric in row:
                            max_val = radar_df[metric].max()
                            if max_val > 0:
                                values.append(row[metric] / max_val * 100)
                            else:
                                values.append(0)
                        else:
                            values.append(0)
                    
                    # Add channel to radar chart
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values + [values[0]],  # Close the loop
                        theta=metrics + [metrics[0]],  # Close the loop
                        fill='toself',
                        name=channel_name
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    title='Channel Performance Comparison (Normalized)'
                )
                
                visualizations['radar_comparison'] = fig_radar
            
            # 4. Sentiment Comparison - Direct comparison between channels
            sentiment_cols = ['Positive Sentiment', 'Neutral Sentiment', 'Negative Sentiment']
            if all(col in raw_df.columns for col in sentiment_cols) and 'Channel Name' in raw_df.columns:
                # Create sentiment comparison dataframe
                sentiment_data = []
                
                for _, row in raw_df.iterrows():
                    channel_name = row['Channel Name']
                    total_sentiment = sum(row[col] for col in sentiment_cols)
                    
                    if total_sentiment > 0:
                        sentiment_data.append({
                            'Channel': channel_name,
                            'Sentiment': 'Positive',
                            'Percentage': (row['Positive Sentiment'] / total_sentiment) * 100
                        })
                        sentiment_data.append({
                            'Channel': channel_name,
                            'Sentiment': 'Neutral',
                            'Percentage': (row['Neutral Sentiment'] / total_sentiment) * 100
                        })
                        sentiment_data.append({
                            'Channel': channel_name,
                            'Sentiment': 'Negative',
                            'Percentage': (row['Negative Sentiment'] / total_sentiment) * 100
                        })
                
                if sentiment_data:
                    sentiment_df = pd.DataFrame(sentiment_data)
                    fig_sentiment = px.bar(
                        sentiment_df,
                        x='Channel',
                        y='Percentage',
                        color='Sentiment',
                        title='Sentiment Distribution Comparison',
                        color_discrete_map={
                            'Positive': '#00CC96',
                            'Neutral': '#636EFA',
                            'Negative': '#EF553B'
                        },
                        text_auto='.1f'
                    )
                    fig_sentiment.update_layout(barmode='stack')
                    visualizations['sentiment_comparison'] = fig_sentiment
                    
                    # Direct sentiment comparison chart (positive sentiment only)
                    positive_sentiment = sentiment_df[sentiment_df['Sentiment'] == 'Positive']
                    fig_positive = px.bar(
                        positive_sentiment,
                        x='Channel',
                        y='Percentage',
                        title='Positive Sentiment Comparison',
                        color='Percentage',
                        color_continuous_scale='Greens',
                        text_auto='.1f'
                    )
                    fig_positive.update_traces(textposition='outside')
                    visualizations['positive_sentiment_comparison'] = fig_positive
            
            # 5. Direct Engagement Metrics Comparison
            if all(col in raw_df.columns for col in ['Channel Name', 'Avg Likes', 'Avg Comments']):
                # Melt the dataframe for comparison
                engagement_df = raw_df[['Channel Name', 'Avg Likes', 'Avg Comments']].melt(
                    id_vars=['Channel Name'],
                    var_name='Metric',
                    value_name='Value'
                )
                
                fig_engagement_metrics = px.bar(
                    engagement_df,
                    x='Channel Name',
                    y='Value',
                    color='Metric',
                    barmode='group',
                    title='Engagement Metrics Comparison',
                    labels={'Value': 'Average Count', 'Channel Name': 'Channel'},
                    color_discrete_map={
                        'Avg Likes': '#FF9900',
                        'Avg Comments': '#9C27B0'
                    }
                )
                visualizations['engagement_metrics_comparison'] = fig_engagement_metrics
            
            logger.info(f"Generated {len(visualizations)} comparison visualizations")
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating comparison visualizations: {e}")
            return {}
    
    def generate_comparison_insights(self, raw_df):
        """Generate direct comparison insights between channels"""
        try:
            if raw_df.empty:
                logger.warning("No data provided for comparison insights")
                return []
            
            insights = []
            
            # 1. Identify top performing channel
            if 'Total Views' in raw_df.columns and 'Channel Name' in raw_df.columns:
                top_views_idx = raw_df['Total Views'].idxmax()
                top_views_channel = raw_df.loc[top_views_idx, 'Channel Name']
                top_views = raw_df.loc[top_views_idx, 'Total Views']
                
                # Find second best for direct comparison
                if len(raw_df) > 1:
                    # Create a copy without the top performer
                    df_without_top = raw_df.drop(top_views_idx)
                    second_views_idx = df_without_top['Total Views'].idxmax()
                    second_views_channel = raw_df.loc[second_views_idx, 'Channel Name']
                    second_views = raw_df.loc[second_views_idx, 'Total Views']
                    
                    # Calculate the difference
                    views_difference = top_views - second_views
                    views_percentage = (views_difference / second_views * 100) if second_views > 0 else 0
                    
                    insights.append({
                        'type': 'direct_comparison',
                        'metric': 'Total Views',
                        'channels': [top_views_channel, second_views_channel],
                        'values': [top_views, second_views],
                        'difference': views_difference,
                        'percentage': views_percentage,
                        'description': f"{top_views_channel} has {int(views_percentage)}% more total views than {second_views_channel} ({int(top_views):,} vs {int(second_views):,})"
                    })
                
                insights.append({
                    'type': 'top_performer',
                    'metric': 'Total Views',
                    'channel': top_views_channel,
                    'value': top_views,
                    'description': f"{top_views_channel} has the highest total views with {int(top_views):,} views"
                })
            
            # 2. Identify channel with best engagement
            if 'Engagement Rate' in raw_df.columns and 'Channel Name' in raw_df.columns:
                top_engagement_idx = raw_df['Engagement Rate'].idxmax()
                top_engagement_channel = raw_df.loc[top_engagement_idx, 'Channel Name']
                top_engagement = raw_df.loc[top_engagement_idx, 'Engagement Rate']
                
                # Find second best for direct comparison
                if len(raw_df) > 1:
                    # Create a copy without the top performer
                    df_without_top = raw_df.drop(top_engagement_idx)
                    second_engagement_idx = df_without_top['Engagement Rate'].idxmax()
                    second_engagement_channel = raw_df.loc[second_engagement_idx, 'Channel Name']
                    second_engagement = raw_df.loc[second_engagement_idx, 'Engagement Rate']
                    
                    # Calculate the difference
                    engagement_difference = top_engagement - second_engagement
                    engagement_percentage = (engagement_difference / second_engagement * 100) if second_engagement > 0 else 0
                    
                    insights.append({
                        'type': 'direct_comparison',
                        'metric': 'Engagement Rate',
                        'channels': [top_engagement_channel, second_engagement_channel],
                        'values': [top_engagement, second_engagement],
                        'difference': engagement_difference,
                        'percentage': engagement_percentage,
                        'description': f"{top_engagement_channel} has {engagement_percentage:.1f}% higher engagement rate than {second_engagement_channel} ({top_engagement:.2f}% vs {second_engagement:.2f}%)"
                    })
                
                insights.append({
                    'type': 'top_performer',
                    'metric': 'Engagement Rate',
                    'channel': top_engagement_channel,
                    'value': top_engagement,
                    'description': f"{top_engagement_channel} has the highest engagement rate at {top_engagement:.2f}%"
                })
            
            # 3. Compare sentiment
            sentiment_cols = ['Positive Sentiment', 'Neutral Sentiment', 'Negative Sentiment']
            if all(col in raw_df.columns for col in sentiment_cols) and 'Channel Name' in raw_df.columns:
                sentiment_comparison = []
                
                for _, row in raw_df.iterrows():
                    channel_name = row['Channel Name']
                    total_sentiment = sum(row[col] for col in sentiment_cols)
                    
                    if total_sentiment > 0:
                        positive_pct = (row['Positive Sentiment'] / total_sentiment) * 100
                        negative_pct = (row['Negative Sentiment'] / total_sentiment) * 100
                        
                        sentiment_comparison.append({
                            'channel': channel_name,
                            'positive_pct': positive_pct,
                            'negative_pct': negative_pct,
                            'ratio': positive_pct / negative_pct if negative_pct > 0 else float('inf')
                        })
                
                if sentiment_comparison:
                    # Sort by positive to negative ratio
                    sentiment_comparison.sort(key=lambda x: x['ratio'], reverse=True)
                    
                    best = sentiment_comparison[0]
                    
                    insights.append({
                        'type': 'sentiment',
                        'channel': best['channel'],
                        'positive_pct': best['positive_pct'],
                        'negative_pct': best['negative_pct'],
                        'ratio': best['ratio'],
                        'description': f"{best['channel']} has the most positive sentiment with {best['positive_pct']:.1f}% positive comments"
                    })
                    
                    if len(sentiment_comparison) >= 2:
                        best = sentiment_comparison[0]
                        worst = sentiment_comparison[-1]
                        
                        insights.append({
                            'type': 'sentiment_comparison',
                            'channels': [best['channel'], worst['channel']],
                            'values': [best['positive_pct'], worst['positive_pct']],
                            'description': f"{best['channel']} has {best['positive_pct'] - worst['positive_pct']:.1f}% more positive sentiment than {worst['channel']} ({best['positive_pct']:.1f}% vs {worst['positive_pct']:.1f}%)"
                        })
            
            logger.info(f"Generated {len(insights)} comparison insights")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating comparison insights: {e}")
            return []
    
    def generate_comparison_recommendations(self, raw_df, insights):
        """Generate actionable recommendations based on direct channel comparison"""
        try:
            if raw_df.empty:
                logger.warning("No data provided for comparison recommendations")
                return []
            
            recommendations = {}
            
            # 1. Content strategy recommendations
            content_recommendations = []
            
            # Identify best performing content types
            if 'Channel Name' in raw_df.columns:
                top_channel = None
                
                for insight in insights:
                    if insight['type'] == 'top_performer' and insight['metric'] == 'Total Views':
                        top_channel = insight['channel']
                        break
                
                if top_channel:
                    content_recommendations.append({
                        'recommendation': f"Study {top_channel}'s content strategy and video formats",
                        'impact': 'high',
                        'category': 'content_strategy'
                    })
                
                # Add direct comparison recommendations
                for insight in insights:
                    if insight['type'] == 'direct_comparison' and insight['metric'] == 'Total Views':
                        better_channel, worse_channel = insight['channels']
                        content_recommendations.append({
                            'recommendation': f"Analyze what {better_channel} does differently from {worse_channel} in terms of content topics and presentation",
                            'impact': 'high',
                            'category': 'content_strategy'
                        })
            
            # 2. Engagement recommendations
            engagement_recommendations = []
            
            # Identify best engagement practices
            top_engagement_channel = None
            
            for insight in insights:
                if insight['type'] == 'top_performer' and insight['metric'] == 'Engagement Rate':
                    top_engagement_channel = insight['channel']
                    break
            
            if top_engagement_channel:
                engagement_recommendations.append({
                    'recommendation': f"Analyze {top_engagement_channel}'s audience interaction techniques",
                    'impact': 'high',
                    'category': 'audience_engagement'
                })
            
            # Add direct comparison engagement recommendations
            for insight in insights:
                if insight['type'] == 'sentiment_comparison':
                    better_channel, worse_channel = insight['channels']
                    engagement_recommendations.append({
                        'recommendation': f"Examine {better_channel}'s comment section management and community engagement compared to {worse_channel}",
                        'impact': 'medium',
                        'category': 'audience_engagement'
                    })
            
            # 3. Growth recommendations
            growth_recommendations = []
            
            # Add general growth recommendations
            growth_recommendations.append({
                'recommendation': "Cross-promote content between channels to leverage each audience",
                'impact': 'medium',
                'category': 'audience_growth'
            })
            
            growth_recommendations.append({
                'recommendation': "Identify content gaps that none of the compared channels are addressing",
                'impact': 'high',
                'category': 'content_strategy'
            })
            
            # Add competitive advantage recommendations
            if len(raw_df) >= 2:
                growth_recommendations.append({
                    'recommendation': "Create content that directly compares or contrasts topics covered by the analyzed channels",
                    'impact': 'high',
                    'category': 'content_strategy'
                })
                
                # Find channels with complementary strengths
                top_views_channel = None
                top_engagement_channel = None
                
                for insight in insights:
                    if insight['type'] == 'top_performer':
                        if insight['metric'] == 'Total Views':
                            top_views_channel = insight['channel']
                        elif insight['metric'] == 'Engagement Rate':
                            top_engagement_channel = insight['channel']
                
                if top_views_channel and top_engagement_channel and top_views_channel != top_engagement_channel:
                    growth_recommendations.append({
                        'recommendation': f"Combine {top_views_channel}'s reach strategies with {top_engagement_channel}'s engagement tactics for optimal growth",
                        'impact': 'high',
                        'category': 'audience_growth'
                    })
            
            # 4. Combine all recommendations
            recommendations = {
                'content': content_recommendations,
                'engagement': engagement_recommendations,
                'growth': growth_recommendations
            }
            
            logger.info(f"Generated comparison recommendations in {len(recommendations)} categories")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating comparison recommendations: {e}")
            return {'content': [], 'engagement': [], 'growth': []}

# --- Helper Functions ---
def verify_youtube_api_key():
    if not youtube_api_key: return False, "❌ Key Missing"
    try: 
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        r = youtube.search().list(part='id', q='test', maxResults=1).execute()
        return True, "✅ API Connected."
    except HttpError as e: 
        logger.error(f"YT key verify failed: {e}")
        return False, f"❌ YouTube API Key Error: Status {e.resp.status}. Check quota/key."
    except Exception as e: 
        logger.error(f"YT key verify error: {e}")
        return False, f"❌ YouTube API Verification Error: {str(e)}"

def batch(iterable, size): 
    iterator = iter(iterable)
    return iter(lambda: list(islice(iterator, size)), [])

def format_value(metric_type, value):
    """Format numeric values for display."""
    try:
        if metric_type in ['Views', 'Likes', 'Comments']:
            return f"{int(value):,}"
        elif metric_type == 'Rate':
            return f"{float(value):.2f}%"
        else:
            return str(value)
    except (ValueError, TypeError):
        return "N/A"

def generate_basic_analysis_fallback(analysis_data):
    """Generate basic analysis when Gemini analysis fails."""
    try:
        formatted_analysis = []
        
        # Basic Performance Overview
        formatted_analysis.append("### 🎯 Overall Performance")
        formatted_analysis.append("*Basic analysis mode due to processing limitations.*")
        
        # Add available metrics
        if 'total_metrics' in analysis_data:
            metrics = analysis_data['total_metrics']
            formatted_analysis.append(f"- Total Views: {format_value('Views', metrics.get('views', 0))}")
            formatted_analysis.append(f"- Total Likes: {format_value('Likes', metrics.get('likes', 0))}")
            formatted_analysis.append(f"- Total Comments: {format_value('Comments', metrics.get('comments', 0))}")

        # Add available averages
        if 'averages' in analysis_data:
            avgs = analysis_data['averages']
            formatted_analysis.append("\n### 📊 Average Performance")
            formatted_analysis.append(f"- Average Views: {format_value('Views', avgs.get('views', 0))}")
            formatted_analysis.append(f"- Average Likes: {format_value('Likes', avgs.get('likes', 0))}")
            formatted_analysis.append(f"- Average Comments: {format_value('Comments', avgs.get('comments', 0))}")
            formatted_analysis.append(f"- Engagement Rate: {format_value('Rate', avgs.get('engagement_rate', 0))}")

        # Add basic recommendations
        formatted_analysis.append("\n### 💡 Basic Recommendations")
        formatted_analysis.append("- Review your content strategy and posting schedule")
        formatted_analysis.append("- Engage with your audience through comments")
        formatted_analysis.append("- Optimize your video titles and thumbnails")
        formatted_analysis.append("- Consider your most viewed content for future video ideas")

        return "\n\n".join(formatted_analysis)
    except Exception as e:
        logger.error(f"Error in basic analysis fallback: {e}")
        return "Error: Could not generate analysis."

def analyze_videos_with_gemini(videos, analysis_type):
    """Analyze videos using Google's Gemini API."""
    try:
        if model is None:
            return "Error: Gemini model not initialized"

        # Prepare video data for analysis
        video_data = []
        for video in videos:
            stats = video.get('statistics', {})
            snippet = video.get('snippet', {})
            video_data.append({
                'title': snippet.get('title', ''),
                'description': snippet.get('description', ''),
                'views': int(stats.get('viewCount', 0)),
                'likes': int(stats.get('likeCount', 0)),
                'comments': int(stats.get('commentCount', 0)),
                'published_at': snippet.get('publishedAt', '')
            })

        # Create a prompt for Gemini
        prompt = f"""As a YouTube channel analysis expert, analyze the following YouTube channel data and provide a detailed summary:

Channel Summary:
- Number of videos: {len(videos)}
- Total views: {sum(v['views'] for v in video_data)}
- Total likes: {sum(v['likes'] for v in video_data)}
- Total comments: {sum(v['comments'] for v in video_data)}

Video Details:
{chr(10).join(f'- Title: {v["title"]} | Views: {v["views"]:,} | Likes: {v["likes"]:,} | Comments: {v["comments"]:,}' for v in video_data)}

Please provide a comprehensive analysis including:
1. Overall channel performance
2. Content strategy insights
3. Engagement patterns
4. Growth opportunities
5. Recommendations for improvement

Format the analysis in a clear, structured way with sections and bullet points where appropriate."""

        # Call Gemini API with retry logic and streaming for faster response
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.7,
                        'candidate_count': 1,
                    },
                    stream=True  # Enable streaming for faster response
                )
                
                # Collect the streamed response
                full_response = []
                for chunk in response:
                    if chunk.text:
                        full_response.append(chunk.text)
                
                if full_response:
                    return "".join(full_response)
                    
                time.sleep(1)  # Add a small delay between retries
            except Exception as retry_error:
                logger.warning(f"Gemini API retry {attempt + 1}/{max_retries}: {retry_error}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

        return "Error: No analysis generated after retries"

    except Exception as e:
        logger.error(f"Error in Gemini analysis: {e}")
        return f"Error generating analysis: {str(e)}"

def enhance_sentiment_analysis_with_gemini(sentiment_data, video_sentiments, channel_name=""):
    """Use Gemini to enhance and provide deeper insights from sentiment analysis data"""
    try:
        if model is None:
            logger.warning("Gemini model not initialized for sentiment enhancement")
            return sentiment_data, video_sentiments, None
        
        # Prepare sentiment data for Gemini analysis
        total_sentiments = sum(sentiment_data.values()) if sentiment_data else 0
        sentiment_percentages = {}
        
        if total_sentiments > 0:
            sentiment_percentages = {
                'positive': (sentiment_data.get('POSITIVE', 0) / total_sentiments) * 100,
                'neutral': (sentiment_data.get('NEUTRAL', 0) / total_sentiments) * 100,
                'negative': (sentiment_data.get('NEGATIVE', 0) / total_sentiments) * 100
            }
        
        # Extract video sentiment data
        video_sentiment_summary = []
        for video in video_sentiments:
            video_sentiment_summary.append({
                'title': video['title'],
                'positive': video['sentiment_percentages'].get('POSITIVE', 0),
                'neutral': video['sentiment_percentages'].get('NEUTRAL', 0),
                'negative': video['sentiment_percentages'].get('NEGATIVE', 0),
                'comments_analyzed': video['analyzed_comment_count']
            })
        
        # Create prompt for Gemini
        channel_context = f" for channel '{channel_name}'" if channel_name else ""
        
        prompt = f"""As a YouTube sentiment analysis expert, analyze the following sentiment data{channel_context} and provide detailed insights:

Overall Sentiment Distribution:
- Positive: {sentiment_percentages.get('positive', 0):.1f}%
- Neutral: {sentiment_percentages.get('neutral', 0):.1f}%
- Negative: {sentiment_percentages.get('negative', 0):.1f}%

Video-specific Sentiment:
{chr(10).join(f"- {v['title'][:50]}...: Positive {v['positive']:.1f}%, Neutral {v['neutral']:.1f}%, Negative {v['negative']:.1f}% (Based on {v['comments_analyzed']} comments)" for v in video_sentiment_summary[:5])}

Please provide:
1. A detailed analysis of the sentiment patterns
2. Insights into what might be driving positive and negative sentiment
3. Actionable recommendations for improving audience sentiment
4. Identification of any concerning sentiment trends that need attention
5. Comparison of sentiment across videos (if multiple videos are present)

Format your analysis in a clear, structured way with sections and bullet points where appropriate."""

        # Call Gemini API with retry logic
        max_retries = 3
        enhanced_analysis = None
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.7,
                        'candidate_count': 1,
                    }
                )
                
                if response.text:
                    enhanced_analysis = response.text
                    break
                    
                time.sleep(1)  # Add a small delay between retries
            except Exception as retry_error:
                logger.warning(f"Gemini API retry for sentiment enhancement {attempt + 1}/{max_retries}: {retry_error}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to enhance sentiment analysis after {max_retries} attempts")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return sentiment_data, video_sentiments, enhanced_analysis
        
    except Exception as e:
        logger.error(f"Error enhancing sentiment analysis with Gemini: {e}")
        return sentiment_data, video_sentiments, None

def display_sentiment_analysis(sentiment_data, video_sentiments, figures_dict, title_prefix=""):
    """Display sentiment analysis results in a formatted way"""
    st.header("💬 Sentiment Analysis")
    
    if not sentiment_data and not video_sentiments:
        st.info("No sentiment data available for analysis.")
        return

    # Get enhanced sentiment analysis from Gemini
    _, _, enhanced_analysis = enhance_sentiment_analysis_with_gemini(
        sentiment_data, 
        video_sentiments,
        title_prefix
    )

    # Overall Sentiment Distribution
    if sentiment_data:
        total_sentiments = sum(sentiment_data.values())
        if total_sentiments > 0:
            st.subheader("Overall Sentiment Distribution")
            cols = st.columns(3)
            
            # Calculate percentages
            pos_pct = (sentiment_data.get('POSITIVE', 0) / total_sentiments) * 100
            neu_pct = (sentiment_data.get('NEUTRAL', 0) / total_sentiments) * 100
            neg_pct = (sentiment_data.get('NEGATIVE', 0) / total_sentiments) * 100
            
            # Display in columns with emojis
            with cols[0]:
                st.metric("😊 Positive", f"{pos_pct:.1f}%")
            with cols[1]:
                st.metric("😐 Neutral", f"{neu_pct:.1f}%")
            with cols[2]:
                st.metric("☹️ Negative", f"{neg_pct:.1f}%")

    # Video-specific Analysis
    if video_sentiments:
        st.subheader("Video-specific Sentiment Analysis")
        
        # Sort videos by positive sentiment percentage
        sorted_videos = sorted(video_sentiments, 
                             key=lambda x: x['sentiment_percentages'].get('POSITIVE', 0),
                             reverse=True)
        
        # Display top 3 most positively received videos
        st.markdown("#### 🌟 Most Positively Received Videos")
        for video in sorted_videos[:3]:
            pos_pct = video['sentiment_percentages'].get('POSITIVE', 0)
            st.markdown(f"""
            **{video['title'][:80]}{'...' if len(video['title']) > 80 else ''}**
            - Positive Sentiment: {pos_pct:.1f}%
            - Total Comments Analyzed: {video['analyzed_comment_count']}
            """)
        
        # Display engagement metrics
        st.subheader("📊 Comment Engagement Metrics")
        total_comments = sum(v['actual_comment_count'] for v in video_sentiments)
        avg_comments = total_comments / len(video_sentiments) if video_sentiments else 0
        
        cols = st.columns(2)
        with cols[0]:
            st.metric("Total Comments", f"{total_comments:,}")
        with cols[1]:
            st.metric("Average Comments per Video", f"{avg_comments:.0f}")

        # Create and display sentiment trend chart
        sentiment_data = []
        for video in video_sentiments:
            sentiment_data.append({
                'Title': video['title'][:30] + '...',
                'Positive': video['sentiment_percentages'].get('POSITIVE', 0),
                'Neutral': video['sentiment_percentages'].get('NEUTRAL', 0),
                'Negative': video['sentiment_percentages'].get('NEGATIVE', 0)
            })
        
        if sentiment_data:
            df = pd.DataFrame(sentiment_data)
            fig = px.bar(df, x='Title', y=['Positive', 'Neutral', 'Negative'],
                        title='Sentiment Distribution Across Videos',
                        labels={'value': 'Percentage', 'variable': 'Sentiment'},
                        color_discrete_map={
                            'Positive': '#00CC96',
                            'Neutral': '#636EFA',
                            'Negative': '#EF553B'
                        })
            st.plotly_chart(fig, use_container_width=True)
            figures_dict['sentiment_distribution'] = fig
    
    # Display enhanced Gemini analysis if available
    if enhanced_analysis:
        st.subheader("🧠 Enhanced Sentiment Analysis")
        st.markdown(enhanced_analysis)
    else:
        # Display basic sentiment insights if Gemini analysis is not available
        if sentiment_data and total_sentiments > 0:
            st.subheader("📊 Basic Sentiment Insights")
            
            # Determine dominant sentiment
            sentiments = {
                'POSITIVE': (sentiment_data.get('POSITIVE', 0) / total_sentiments) * 100,
                'NEUTRAL': (sentiment_data.get('NEUTRAL', 0) / total_sentiments) * 100,
                'NEGATIVE': (sentiment_data.get('NEGATIVE', 0) / total_sentiments) * 100
            }
            
            dominant_sentiment = max(sentiments, key=sentiments.get)
            
            if dominant_sentiment == 'POSITIVE':
                st.success(f"The audience sentiment is predominantly positive ({sentiments['POSITIVE']:.1f}%), indicating good audience reception.")
                st.markdown("**Recommendations:**")
                st.markdown("- Continue with similar content that resonates with your audience")
                st.markdown("- Engage with positive comments to reinforce audience connection")
                st.markdown("- Consider highlighting positive audience feedback in future videos")
            elif dominant_sentiment == 'NEUTRAL':
                st.info(f"The audience sentiment is predominantly neutral ({sentiments['NEUTRAL']:.1f}%), suggesting room for more engaging content.")
                st.markdown("**Recommendations:**")
                st.markdown("- Create more emotionally engaging content to drive stronger reactions")
                st.markdown("- Ask more direct questions to your audience to encourage engagement")
                st.markdown("- Experiment with different content formats to find what resonates better")
            else:
                st.error(f"The audience sentiment is predominantly negative ({sentiments['NEGATIVE']:.1f}%), indicating potential issues to address.")
                st.markdown("**Recommendations:**")
                st.markdown("- Review negative comments to identify specific concerns")
                st.markdown("- Address audience feedback directly in future videos")
                st.markdown("- Consider adjusting your content approach based on criticism")

def display_unified_comparison_analysis(comparison_df_formatted, comparison_df_raw, channel_info, all_channel_results):
    """Display unified comparison analysis results in a single view"""
    st.header("📊 Channel Comparison Analysis")
    
    if comparison_df_formatted.empty:
        st.info("No comparison data available.")
        return
    
    # Initialize the enhanced comparison analyzer
    comparison_analyzer = EnhancedChannelComparison()
    
    # Generate visualizations
    visualizations = comparison_analyzer.generate_comparison_visualizations(comparison_df_raw)
    
    # Generate insights
    insights = comparison_analyzer.generate_comparison_insights(comparison_df_raw)
    
    # Generate recommendations
    recommendations = comparison_analyzer.generate_comparison_recommendations(comparison_df_raw, insights)
    
    # Display unified comparison metrics
    st.subheader("📈 Performance Metrics Comparison")
    
    # Create a unified metrics table
    metrics_data = []
    for _, row in comparison_df_raw.iterrows():
        channel_name = row['Channel Name']
        metrics_data.append({
            'Channel': channel_name,
            'Total Views': int(row['Total Views']),
            'Total Likes': int(row['Total Likes']),
            'Total Comments': int(row['Total Comments']),
            'Engagement Rate': f"{row['Engagement Rate']:.2f}%",
            'Videos Analyzed': int(row['Videos Analyzed'])
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Display key visualizations
    st.subheader("📊 Visual Comparison")
    
    # Display total views comparison
    if 'total_views' in visualizations:
        st.plotly_chart(visualizations['total_views'], use_container_width=True)
    
    # Display engagement rate comparison
    if 'engagement_rate' in visualizations:
        st.plotly_chart(visualizations['engagement_rate'], use_container_width=True)
    
    # Display radar chart comparison
    if 'radar_comparison' in visualizations:
        st.plotly_chart(visualizations['radar_comparison'], use_container_width=True)
    
    # Display sentiment comparison if available
    if 'sentiment_comparison' in visualizations:
        st.plotly_chart(visualizations['sentiment_comparison'], use_container_width=True)
    
    # Display engagement metrics comparison
    if 'engagement_metrics_comparison' in visualizations:
        st.plotly_chart(visualizations['engagement_metrics_comparison'], use_container_width=True)
    
    # Display unified insights
    if insights:
        st.subheader("🔍 Key Insights")
        
        # Group insights by type
        top_performers = [i for i in insights if i['type'] == 'top_performer']
        direct_comparisons = [i for i in insights if i['type'] == 'direct_comparison']
        sentiment_insights = [i for i in insights if i['type'] in ['sentiment', 'sentiment_comparison']]
        
        # Display all insights in a single section
        for insight in top_performers:
            st.markdown(f"🏆 **{insight['description']}**")
        
        for insight in direct_comparisons:
            st.markdown(f"📊 **{insight['description']}**")
        
        for insight in sentiment_insights:
            st.markdown(f"💬 **{insight['description']}**")
    
    # Display unified recommendations
    if recommendations:
        st.subheader("💡 Strategic Recommendations")
        
        # Display all recommendations in a single section
        for category, recs in recommendations.items():
            if recs:
                st.markdown(f"**{category.title()} Recommendations:**")
                for rec in recs:
                    impact_emoji = "🔥" if rec['impact'] == 'high' else "✨"
                    st.markdown(f"{impact_emoji} {rec['recommendation']}")
    
    # Display unified video comparison
    st.subheader("📺 Top Videos Comparison")
    
    # Create a unified video dataframe
    all_videos = []
    for channel_id, results in all_channel_results.items():
        channel_name = channel_info.get(channel_id, {}).get('title', channel_id)
        channel_df = results.get('dataframe', pd.DataFrame())
        
        if not channel_df.empty:
            # Get top 3 videos by views
            top_videos = channel_df.sort_values('Views', ascending=False).head(3)
            
            for _, video in top_videos.iterrows():
                all_videos.append({
                    'Channel': channel_name,
                    'Title': video['Title'],
                    'Views': video['Views'],
                    'Likes': video['Likes'],
                    'Comments': video['Comments'],
                    'Published': video['Published']
                })
    
    if all_videos:
        videos_df = pd.DataFrame(all_videos)
        videos_df = videos_df.sort_values('Views', ascending=False)
        st.dataframe(videos_df, use_container_width=True)
    else:
        st.info("No video data available for comparison.")

# --- Main Streamlit Application ---
def main():
    st.title("📊 YouTube Channel Analyzer")
    youtube_key_valid, youtube_status_msg = verify_youtube_api_key()
    gemini_key_valid = bool(gemini_api_key)
    
    # Display API status
    st.markdown(f"""<div style='border: 1px solid #e0e0e0; border-radius: 5px; padding: 10px; margin-bottom: 10px;'>
        <h6 style='margin-bottom:5px'><span style='font-size: 1.2em;'>{'✅' if youtube_key_valid else '❌'}</span> YouTube API</h6>
        <small>{youtube_status_msg}</small>
        <h6 style='margin-bottom:5px; margin-top: 10px;'><span style='font-size: 1.2em;'>{'✅' if gemini_key_valid else '❌'}</span> Gemini API</h6>
        <small>{"API Key configured" if gemini_key_valid else "API Key missing"}</small>
    </div>""", unsafe_allow_html=True)
    st.divider()
    
    # --- Analysis Mode Selection ---
    st.header("Select Analysis Mode")
    analysis_mode = st.radio(
        "Choose analysis mode:",
        ("Single Channel Analysis", "Multiple Channel Comparison"),
        index=0,
        help="Select whether to analyze a single channel or compare multiple channels"
    )
    
    st.divider()
    
    # --- Controls Section ---
    st.header("Analysis Settings")
    st.markdown("---")
    channel_urls_input = st.text_area(
        "Enter YouTube Channel URLs", 
        placeholder="Enter one YouTube channel URL per line...", 
        height=100,
        help="For multiple channel comparison, enter at least 2 channel URLs"
    )
    
    # Show warning if not enough channels for comparison mode
    if analysis_mode == "Multiple Channel Comparison":
        urls = [url.strip() for url in channel_urls_input.split('\n') if url.strip()]
        if len(urls) < 2:
            st.warning("Please enter at least 2 channel URLs for comparison analysis.")
    
    st.markdown("###### Date Range")
    date_col_1, date_col_2 = st.columns(2)
    default_end_date = datetime.now().date()
    default_start_date = default_end_date - timedelta(days=30)
    with date_col_1:
        start_date_input = st.date_input("Start Date", default_start_date)
    with date_col_2:
        end_date_input = st.date_input("End Date", default_end_date)
    valid_date_range = start_date_input <= end_date_input
    if not valid_date_range:
        st.error("Start date cannot be after end date.")
    
    st.info("⚠️ API Quota: Each analysis consumes YouTube API quota. Multiple channels will use more quota.", icon="⚠️")
    
    # Disable analyze button if not enough channels for comparison
    button_disabled = not youtube_key_valid or not valid_date_range
    if analysis_mode == "Multiple Channel Comparison":
        urls = [url.strip() for url in channel_urls_input.split('\n') if url.strip()]
        if len(urls) < 2:
            button_disabled = True
    
    col1, col2 = st.columns(2)
    with col1:
        max_videos_per_channel = st.slider("Max Videos per Channel", 1, 100, 10)
    with col2:
        max_comments = st.slider("Comments Per Video", 0, 100, 10)
    
    st.markdown("###### Analysis Type")
    analysis_options = {
        "Channel Summary": "Gemini Analysis",
        "Comment Analysis": "Sentiment Analysis",
        "Complete Analysis": "Gemini + Sentiment"
    }
    default_analysis_ix = 0 if analysis_mode == "Multiple Channel Comparison" else 2
    selected_analysis_type_key = st.radio(
        "Analysis Type", 
        list(analysis_options.keys()), 
        index=default_analysis_ix, 
        key="analysis_type_radio"
    )
    st.caption(analysis_options[selected_analysis_type_key])
    
    analyze_button = st.button(
        "Analyze", 
        use_container_width=True, 
        type="primary", 
        key="analyze_btn", 
        disabled=button_disabled
    )
    
    # --- Results Section ---
    session_keys = [
        'analysis_complete', 'all_channel_results', 'channel_info', 
        'aggregate_results', 'aggregate_figures', 'aggregate_videos', 
        'comparison_df_formatted', 'comparison_df_raw', 'current_analysis_mode'
    ]
    
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = {} if 'results' in key or 'info' in key or 'figures' in key else (
                pd.DataFrame() if 'df' in key else (
                    [] if 'videos' in key else (
                        False if key == 'analysis_complete' else None
                    )
                )
            )

    if analyze_button:
        # Reset session state
        for key in session_keys:
            st.session_state[key] = {} if 'results' in key or 'info' in key or 'figures' in key else (
                pd.DataFrame() if 'df' in key else (
                    [] if 'videos' in key else (
                        False if key == 'analysis_complete' else None
                    )
                )
            )
        
        st.session_state.current_analysis_mode = analysis_mode
        proceed_with_analysis = True
        urls = []
        
        if not channel_urls_input:
            st.warning("Please enter URLs.")
            proceed_with_analysis = False
        else:
            urls = [url.strip() for url in channel_urls_input.split('\n') if url.strip()]
        
        if not urls:
            st.warning("No valid URLs.")
            proceed_with_analysis = False
        
        if analysis_mode == "Multiple Channel Comparison" and len(urls) < 2:
            st.warning("Please enter at least 2 channel URLs for comparison analysis.")
            proceed_with_analysis = False
        
        if not valid_date_range:
            st.error("Invalid date range.")
            proceed_with_analysis = False

        if proceed_with_analysis:
            analysis_type_to_run = selected_analysis_type_key
            runtime_max_comments = max_comments if analysis_type_to_run != "Channel Summary" else 0
            status_placeholder = st.empty()
            status_placeholder.info("✨ Starting analysis...")
            
            with st.spinner(f"🔬 Analyzing ({analysis_mode})..."):
                try:
                    # ================= MULTIPLE CHANNEL COMPARISON =================
                    if analysis_mode == "Multiple Channel Comparison":
                        status_placeholder.info("Fetching videos for multiple channels...")
                        videos_by_channel_raw, channel_info_fetched = get_youtube_videos_grouped(urls, max_videos_per_channel, status_placeholder)
                        st.session_state.channel_info = channel_info_fetched
                        
                        if not videos_by_channel_raw:
                            status_placeholder.warning("No videos found.")
                            st.stop()
                        
                        status_placeholder.info("Filtering videos by date...")
                        start_date_str = start_date_input.strftime('%Y-%m-%d')
                        end_date_str = end_date_input.strftime('%Y-%m-%d')
                        videos_by_channel_filtered = filter_videos_by_date_grouped(videos_by_channel_raw, start_date_str, end_date_str)
                        
                        if not videos_by_channel_filtered:
                            status_placeholder.warning("No videos in date range.")
                            st.stop()
                        
                        status_placeholder.info(f"Analyzing {len(videos_by_channel_filtered)} channels...")
                        temp_results = {}
                        
                        for channel_id, channel_videos in videos_by_channel_filtered.items():
                            channel_title = st.session_state.channel_info.get(channel_id, {}).get('title', channel_id)
                            status_placeholder.info(f"Analyzing: {channel_title}...")
                            
                            channel_results = {'num_videos': len(channel_videos), 'figures': {}}
                            channel_df = pd.DataFrame()
                            
                            try:
                                channel_df = pd.DataFrame([{
                                    'Vid': v['id'],
                                    'Title': v['snippet']['title'],
                                    'Views': int(v['statistics'].get('viewCount', 0)),
                                    'Likes': int(v['statistics'].get('likeCount', 0)),
                                    'Comments': int(v['statistics'].get('commentCount', 0)),
                                    'Published': pd.to_datetime(v['snippet']['publishedAt']).date()
                                } for v in channel_videos])
                            except Exception:
                                pass
                            
                            if not channel_df.empty:
                                channel_df = channel_df.sort_values('Published')
                                channel_results['total_metrics'] = {
                                    'views': channel_df['Views'].sum(),
                                    'likes': channel_df['Likes'].sum(),
                                    'comments': channel_df['Comments'].sum()
                                }
                                channel_results['averages'] = {
                                    'views': channel_df['Views'].mean(),
                                    'likes': channel_df['Likes'].mean(),
                                    'comments': channel_df['Comments'].mean()
                                }
                            
                            channel_results.setdefault('total_metrics', {})
                            channel_results.setdefault('averages', {})

                            if analysis_type_to_run != "Channel Summary" and runtime_max_comments > 0:
                                try:
                                    status_placeholder.info(f"Analyzing sentiment for: {channel_title}...")
                                    sentiment_data = analyze_channel_performance(channel_videos, runtime_max_comments)
                                except APIQuotaExceeded:
                                    raise
                                except Exception as se:
                                    logger.error(f"Sent err {channel_title}: {se}")
                                    sentiment_data = None
                                
                                if sentiment_data:
                                    channel_results.update(sentiment_data)
                            
                            channel_results.setdefault('sentiment_analysis', {})
                            channel_results.setdefault('video_sentiments', [])

                            if analysis_type_to_run != "Comment Analysis":
                                try:
                                    status_placeholder.info(f"Running Gemini analysis for: {channel_title}...")
                                    gemini_analysis = analyze_videos_with_gemini(channel_videos, analysis_type_to_run)
                                    channel_results['gemini_analysis'] = gemini_analysis
                                except Exception as mle:
                                    logger.error(f"Analysis err {channel_title}: {mle}")
                                    channel_results['gemini_analysis'] = generate_basic_analysis_fallback(channel_results)
                            else:
                                channel_results['gemini_analysis'] = "Analysis skipped."

                            channel_results['dataframe'] = channel_df
                            channel_results['videos_list'] = channel_videos
                            temp_results[channel_id] = channel_results

                        st.session_state.all_channel_results = temp_results
                        
                        # Create comparison dataframes
                        comparison_analyzer = EnhancedChannelComparison()
                        comparison_df_formatted, comparison_df_raw = comparison_analyzer.compare_channels(
                            st.session_state.all_channel_results, 
                            st.session_state.channel_info
                        )
                        
                        st.session_state.comparison_df_formatted = comparison_df_formatted
                        st.session_state.comparison_df_raw = comparison_df_raw
                        
                        st.session_state.analysis_complete = True
                        status_placeholder.empty()

                    # ================= SINGLE CHANNEL ANALYSIS =================
                    else:
                        status_placeholder.info("Fetching videos for single channel...")
                        videos_by_channel_raw, channel_info_fetched = get_youtube_videos_grouped(urls, max_videos_per_channel, status_placeholder)
                        all_videos_raw = [video for vid_list in videos_by_channel_raw.values() for video in vid_list]
                        st.session_state.channel_info = channel_info_fetched
                        
                        if not all_videos_raw:
                            status_placeholder.warning("No videos found.")
                            st.stop()
                        
                        status_placeholder.info(f"Fetched {len(all_videos_raw)} total videos. Filtering...")
                        start_date_str = start_date_input.strftime('%Y-%m-%d')
                        end_date_str = end_date_input.strftime('%Y-%m-%d')

                        def filter_agg(vids, s, e):
                            filtered_list = []
                            try:
                                sd = datetime.strptime(s, '%Y-%m-%d')
                                ed = datetime.strptime(e, '%Y-%m-%d') + timedelta(days=1)
                            except ValueError:
                                logger.error("Invalid start/end date format provided to filter_agg")
                                return vids
                            for v in vids:
                                p = v.get('snippet', {}).get('publishedAt')
                                if not p:
                                    continue
                                try:
                                    pd_dt = datetime.fromisoformat(p.replace('Z', '+00:00'))
                                    if sd <= pd_dt.replace(tzinfo=None) < ed:
                                        filtered_list.append(v)
                                except ValueError:
                                    logger.warning(f"Skipping video {v.get('id', 'N/A')} - could not parse date '{p}'")
                                    continue
                            return filtered_list

                        filtered_videos = filter_agg(all_videos_raw, start_date_str, end_date_str)
                        st.session_state.aggregate_videos = filtered_videos
                        
                        if not filtered_videos:
                            status_placeholder.warning("No videos in date range.")
                            st.stop()
                        
                        status_placeholder.info(f"Analyzing {len(filtered_videos)} videos...")
                        temp_results = {}
                        temp_figures = {}
                        agg_df = pd.DataFrame()
                        
                        try:
                            agg_df = pd.DataFrame([{
                                'Vid': v['id'],
                                'Title': v['snippet']['title'],
                                'Views': int(v['statistics'].get('viewCount', 0)),
                                'Likes': int(v['statistics'].get('likeCount', 0)),
                                'Comments': int(v['statistics'].get('commentCount', 0)),
                                'Published': pd.to_datetime(v['snippet']['publishedAt']).date()
                            } for v in filtered_videos])
                            
                            if not agg_df.empty:
                                agg_df = agg_df.sort_values('Published')
                                temp_results['total_metrics'] = {
                                    'views': agg_df['Views'].sum(),
                                    'likes': agg_df['Likes'].sum(),
                                    'comments': agg_df['Comments'].sum()
                                }
                                temp_results['averages'] = {
                                    'views': agg_df['Views'].mean(),
                                    'likes': agg_df['Likes'].mean(),
                                    'comments': agg_df['Comments'].mean()
                                }
                        except Exception as df_ex:
                            logger.error(f"Agg DF Error: {df_ex}")
                        
                        temp_results.setdefault('total_metrics', {})
                        temp_results.setdefault('averages', {})
                        temp_results['num_videos'] = len(filtered_videos)
                        temp_results['dataframe'] = agg_df

                        if analysis_type_to_run != "Channel Summary" and runtime_max_comments > 0:
                            status_placeholder.info("Analyzing sentiment...")
                            try:
                                sentiment_data = analyze_channel_performance(filtered_videos, runtime_max_comments)
                            except APIQuotaExceeded:
                                raise
                            except Exception as se:
                                logger.error(f"Agg Sent err: {se}")
                                sentiment_data = None
                            
                            if sentiment_data:
                                temp_results.update(sentiment_data)
                        
                        temp_results.setdefault('sentiment_analysis', {})
                        temp_results.setdefault('video_sentiments', [])

                        if analysis_type_to_run != "Comment Analysis":
                            status_placeholder.info("Running Gemini analysis...")
                            try:
                                gemini_analysis = analyze_videos_with_gemini(filtered_videos, analysis_type_to_run)
                                temp_results['gemini_analysis'] = gemini_analysis
                            except Exception as mle:
                                logger.error(f"Agg analysis err: {mle}")
                                temp_results['gemini_analysis'] = generate_basic_analysis_fallback(temp_results)
                        else:
                            temp_results['gemini_analysis'] = "Analysis skipped."

                        st.session_state.aggregate_results = temp_results
                        st.session_state.aggregate_figures = temp_figures
                        st.session_state.analysis_complete = True
                        status_placeholder.empty()

                except APIQuotaExceeded as qem:
                    status_placeholder.empty()
                    st.error(f"API Quota Exceeded: {qem}. Analysis halted.")
                except (ConnectionError, ValueError, RuntimeError) as specific_err:
                    status_placeholder.empty()
                    st.error(f"Analysis Error: {specific_err}")
                except Exception as e:
                    status_placeholder.empty()
                    st.error(f"An unexpected error occurred during analysis: {str(e)}")
                    logger.exception("Unhandled main error")
                finally:
                    if status_placeholder:
                        status_placeholder.empty()

    # --- Display Area ---
    if st.session_state.analysis_complete:
        st.divider()
        mode_run = st.session_state.current_analysis_mode
        analysis_type_run = selected_analysis_type_key

        # ================= Display Multiple Channel Comparison Results =================
        if mode_run == "Multiple Channel Comparison":
            if st.session_state.all_channel_results:
                # Display unified comparison analysis
                display_unified_comparison_analysis(
                    st.session_state.comparison_df_formatted,
                    st.session_state.comparison_df_raw,
                    st.session_state.channel_info,
                    st.session_state.all_channel_results
                )
                
                # Display sentiment analysis if available
                if analysis_type_run != "Channel Summary":
                    # Combine sentiment data from all channels
                    combined_sentiment = {}
                    combined_video_sentiments = []
                    
                    for channel_id, results in st.session_state.all_channel_results.items():
                        channel_sentiment = results.get('sentiment_analysis', {})
                        for sentiment_type, count in channel_sentiment.items():
                            combined_sentiment[sentiment_type] = combined_sentiment.get(sentiment_type, 0) + count
                        
                        # Add channel name to video sentiments
                        channel_name = st.session_state.channel_info.get(channel_id, {}).get('title', channel_id)
                        for video_sentiment in results.get('video_sentiments', []):
                            video_sentiment['channel'] = channel_name
                            combined_video_sentiments.append(video_sentiment)
                    
                    if combined_sentiment or combined_video_sentiments:
                        display_sentiment_analysis(
                            combined_sentiment,
                            combined_video_sentiments,
                            {},
                            "All Channels"
                        )
                
                st.divider()
                st.success(f"✅ Multiple Channel Comparison ({analysis_type_run}) completed!")
            else:
                st.warning("Comparison analysis ran, but no results.")

        # ================= Display Single Channel Analysis Results =================
        elif mode_run == "Single Channel Analysis":
            if st.session_state.aggregate_results:
                agg_results = st.session_state.aggregate_results
                agg_df = agg_results.get('dataframe', pd.DataFrame())
                agg_figures = st.session_state.aggregate_figures
                st.header("📊 Performance Overview")

                # Display metrics in columns
                metric_cols = st.columns(3)
                total_metrics = agg_results.get('total_metrics', {})
                with metric_cols[0]:
                    st.metric("Total Views", format_value("Views", total_metrics.get('views', 0)))
                with metric_cols[1]:
                    st.metric("Total Likes", format_value("Likes", total_metrics.get('likes', 0)))
                with metric_cols[2]:
                    st.metric("Total Comments", format_value("Comments", total_metrics.get('comments', 0)))

                # Display averages
                avg_cols = st.columns(3)
                averages = agg_results.get('averages', {})
                with avg_cols[0]:
                    st.metric("Average Views", format_value("Views", averages.get('views', 0)))
                with avg_cols[1]:
                    st.metric("Average Likes", format_value("Likes", averages.get('likes', 0)))
                with avg_cols[2]:
                    st.metric("Average Comments", format_value("Comments", averages.get('comments', 0)))

                st.divider()
                st.subheader("📈 Performance Trends")

                # Create and display trend charts if we have data
                if not agg_df.empty:
                    # Views over time
                    fig_views = px.line(agg_df, x='Published', y='Views', title='Views Over Time')
                    st.plotly_chart(fig_views, use_container_width=True)

                    # Engagement metrics
                    fig_engagement = px.line(agg_df, x='Published', y=['Likes', 'Comments'], title='Engagement Over Time')
                    st.plotly_chart(fig_engagement, use_container_width=True)

                    # Store figures for PDF
                    agg_figures['views_trend'] = fig_views
                    agg_figures['engagement_trend'] = fig_engagement
                else:
                    st.info("No trend data available for the selected date range.")

                # Display video list
                st.subheader("📺 Videos Analyzed")
                if not agg_df.empty:
                    st.dataframe(agg_df[['Title', 'Views', 'Likes', 'Comments', 'Published']].sort_values('Published', ascending=False),
                               use_container_width=True)
                else:
                    st.info("No videos found in the selected date range.")

                if analysis_type_run != "Channel Summary":
                    display_sentiment_analysis(agg_results.get('sentiment_analysis', {}),
                                             agg_results.get('video_sentiments', []),
                                             agg_figures, "Channel")

                if analysis_type_run != "Comment Analysis":
                    st.divider()
                    st.header("🧠 Gemini Analysis")
                    st.markdown(agg_results.get('gemini_analysis', "No analysis available."))

                st.divider()
                st.success(f"✅ Single Channel Analysis ({analysis_type_run}) completed!")
            else:
                st.warning("Analysis ran, but no results.")
    else:
        st.markdown("""
        ### No Analysis Results Yet
        
        1. Select your analysis mode (Single Channel or Multiple Channel Comparison)
        2. Enter YouTube channel URL(s)
        3. Configure date range and other settings
        4. Click **"Analyze"** to start
        
        For multiple channel comparison, enter at least 2 channel URLs.
        """)

# --- Entry Point ---
if __name__ == "__main__":
    if sentiment_pipeline is None:
        st.warning("Local sentiment model failed. Comment analysis unavailable.", icon="⚠️")
    main()