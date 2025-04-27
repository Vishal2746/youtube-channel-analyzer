# app.py (Corrected filter_agg Logic & Label Warnings)

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
# **** USE FPDF2 ****
from fpdf import FPDF # Make sure fpdf2 is installed (pip install fpdf2)
# **** ****
import matplotlib.pyplot as plt
import plotly.io as pio
import io
import base64
from ml_model import YouTubeAnalysisModel
import tempfile
import re
import time
import numpy as np
from collections import defaultdict
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
load_dotenv(); logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'); logger = logging.getLogger(__name__)

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

# --- Helper Functions ---
def verify_youtube_api_key():
    if not youtube_api_key: return False, "❌ Key Missing"
    try: youtube = build('youtube', 'v3', developerKey=youtube_api_key); r = youtube.search().list(part='id', q='test', maxResults=1).execute(); return True, "✅ API Connected."
    except HttpError as e: logger.error(f"YT key verify failed: {e}"); return False, f"❌ YouTube API Key Error: Status {e.resp.status}. Check quota/key."
    except Exception as e: logger.error(f"YT key verify error: {e}"); return False, f"❌ YouTube API Verification Error: {str(e)}"

def add_download_button(analysis_results, analysis_type, channel_info=None):
    """Add a download button for the PDF report"""
    try:
        # First generate the PDF from the analysis results
        pdf_bytes = create_pdf_report(analysis_results, analysis_type, channel_info)
        
        if pdf_bytes is None:
            st.error("Failed to generate PDF report.")
            return
        
        # Log the type of pdf_bytes for debugging
        logger.info(f"PDF bytes type: {type(pdf_bytes)}")
        
        # Handle different data types
        if isinstance(pdf_bytes, bytearray):
            pdf_bytes = bytes(pdf_bytes)
        elif isinstance(pdf_bytes, dict):
            st.error("PDF generation returned a dictionary instead of binary data. Please check the report generator.")
            logger.error(f"Invalid binary data format: {type(pdf_bytes)}")
            return
        elif not isinstance(pdf_bytes, bytes):
            st.error(f"Unexpected data type for PDF: {type(pdf_bytes)}")
            return
        
        # Create download button
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"youtube_analysis_{analysis_type.lower().replace(' ', '_')}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Error adding download button: {e}")
        logger.error(f"Error adding download button: {e}")

def batch(iterable, size): iterator = iter(iterable); return iter(lambda: list(islice(iterator, size)), [])
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
    """Generate basic analysis when OpenAI analysis fails."""
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
def create_pdf_report(analysis_results, analysis_type, channel_info=None):
    """Create a detailed PDF report using the report generator"""
    try:
        from detailed_report_generator import report_generator
        pdf_bytes = report_generator.create_pdf_report(analysis_results, analysis_type, channel_info)
        return pdf_bytes
    except Exception as e:
        logger.error(f"Error creating PDF report: {e}")
        return None

def create_comparison_dataframe(all_channel_results, channel_info): return pd.DataFrame(), pd.DataFrame() # Stub

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

def display_sentiment_analysis(sentiment_data, video_sentiments, figures_dict, title_prefix=""):
    """Display sentiment analysis results in a formatted way with ML/Gemini enhancements"""
    st.header("💬 Sentiment Analysis")
    
    if not sentiment_data and not video_sentiments:
        st.info("No sentiment data available for analysis.")
        return

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
            
            # Display Gemini's enhanced analysis for each video if available
            if 'enhanced_analysis' in video:
                with st.expander("🧠 Enhanced Analysis"):
                    st.markdown(video['enhanced_analysis'])
        
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

    # Display overall enhanced analysis from Gemini if available
    if hasattr(video_sentiments, 'get') and video_sentiments.get('overall_enhanced_analysis'):
        st.subheader("🧠 Overall Enhanced Analysis")
        st.markdown(video_sentiments['overall_enhanced_analysis'])
    
    # Display sample comments if available
    if any('comments' in v for v in video_sentiments):
        st.subheader("💬 Sample Comments")
        
        for video in video_sentiments:
            if 'comments' in video and video['comments']:
                st.markdown(f"**{video['title'][:50]}...**")
                for comment in video['comments'][:2]:  # Show up to 2 comments per video
                    sentiment_color = "#00CC96" if comment['sentiment'] == 'POSITIVE' else (
                                      "#636EFA" if comment['sentiment'] == 'NEUTRAL' else "#EF553B")
                    
                    st.markdown(f"""
                    <div style="border-left: 4px solid {sentiment_color}; padding-left: 10px; margin-bottom: 10px;">
                        <p style="font-style: italic; font-size: 14px;">{comment['text'][:200]}{'...' if len(comment['text']) > 200 else ''}</p>
                        <p style="font-size: 12px; color: gray;">Sentiment: {comment['sentiment']} (Score: {comment['score']:.2f})</p>
                    </div>
                    """, unsafe_allow_html=True)

# --- Main Streamlit Application ---
def main():
    st.title("📊 YouTube Channel Analyzer")
    youtube_key_valid, youtube_status_msg = verify_youtube_api_key()
    gemini_key_valid = bool(gemini_api_key)
    
    st.divider()
    col_controls, col_results = st.columns([1, 2])

    # --- Controls Column ---
    with col_controls:
        st.header("Analysis Settings")
        analysis_mode = st.radio("Analysis Mode", ("Aggregate Analysis", "Compare Channels"), index=0, horizontal=True)
        st.markdown("---")
        channel_urls_input = st.text_area(" ", placeholder="Enter YouTube channel URLs...", height=100, label_visibility="collapsed")
        st.markdown("###### Date Range")
        date_col_1, date_col_2 = st.columns(2)
        default_end_date = datetime.now().date()
        default_start_date = default_end_date - timedelta(days=30)
        with date_col_1:
            start_date_input = st.date_input(" ", default_start_date, label_visibility="collapsed")
        with date_col_2:
            end_date_input = st.date_input(" ", default_end_date, label_visibility="collapsed")
        valid_date_range = start_date_input <= end_date_input
        if not valid_date_range:
            st.error("Start date cannot be after end date.")
            
        analyze_button = st.button("Analyze", use_container_width=True, type="primary", key="analyze_btn", disabled=not youtube_key_valid or not valid_date_range)
        st.divider()
        max_videos_per_channel = st.slider("Max Videos per Channel", 1, 100, 10, label_visibility="collapsed")
        max_comments = st.slider("Comments Per Video", 0, 100, 10, label_visibility="collapsed")
        st.divider()
        st.markdown("###### Analysis Type")
        analysis_options = {
            "Channel Summary": "Gemini Analysis",
            "Comment Analysis": "Sentiment Analysis",
            "Complete Analysis": "Gemini + Sentiment"
        }
        default_analysis_ix = 0 if analysis_mode == "Compare Channels" else 2
        selected_analysis_type_key = st.radio(" ", list(analysis_options.keys()), index=default_analysis_ix, label_visibility="collapsed", key="analysis_type_radio")
        st.caption(analysis_options[selected_analysis_type_key])

    # --- Results Column ---
    with col_results:
        session_keys = ['analysis_complete', 'all_channel_results', 'channel_info', 'aggregate_results', 'aggregate_figures', 'aggregate_videos', 'comparison_df_formatted', 'comparison_df_raw', 'current_analysis_mode']
        for key in session_keys:
            if key not in st.session_state:
                st.session_state[key] = {} if 'results' in key or 'info' in key or 'figures' in key else (pd.DataFrame() if 'df' in key else ([] if 'videos' in key else (False if key == 'analysis_complete' else None)))

        if analyze_button:
            for key in session_keys:
                st.session_state[key] = {} if 'results' in key or 'info' in key or 'figures' in key else (pd.DataFrame() if 'df' in key else ([] if 'videos' in key else (False if key == 'analysis_complete' else None)))
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
                        # ================= COMPARE MODE =================
                        if analysis_mode == "Compare Channels":
                            status_placeholder.info("Fetching videos (grouped)...")
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
                            st.session_state.analysis_complete = True
                            status_placeholder.empty()

                        # ================= AGGREGATE MODE =================
                        else:
                            status_placeholder.info("Fetching videos (aggregated)...")
                            videos_by_channel_raw, channel_info_fetched = get_youtube_videos_grouped(urls, max_videos_per_channel, status_placeholder)
                            all_videos_raw = [video for vid_list in videos_by_channel_raw.values() for video in vid_list]
                            st.session_state.channel_info = channel_info_fetched
                            if not all_videos_raw:
                                status_placeholder.warning("No videos found.")
                                st.stop()
                            status_placeholder.info(f"Fetched {len(all_videos_raw)} total videos. Filtering...")
                            start_date_str = start_date_input.strftime('%Y-%m-%d')
                            end_date_str = end_date_input.strftime('%Y-%m-%d')

                            def filter_agg(vids, s, e, max_limit):
                                filtered_list = []
                                try:
                                    sd = datetime.strptime(s, '%Y-%m-%d')
                                    ed = datetime.strptime(e, '%Y-%m-%d') + timedelta(days=1)
                                except ValueError:
                                    logger.error("Invalid start/end date format provided to filter_agg")
                                    return vids
                                
                                # Sort videos by publishedAt date in descending order (newest first)
                                sorted_vids = sorted(vids, 
                                                    key=lambda x: x.get('snippet', {}).get('publishedAt', ''),
                                                    reverse=True)
                                
                                for v in sorted_vids:
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
                                
                                return filtered_list[:max_limit]  # Limit to max_videos_per_channel

                            filtered_videos = filter_agg(all_videos_raw, start_date_str, end_date_str, max_videos_per_channel)
                            st.session_state.aggregate_videos = filtered_videos
                            if not filtered_videos:
                                status_placeholder.warning("No videos in date range.")
                                st.stop()
                            status_placeholder.info(f"Analyzing {len(filtered_videos)} videos (aggregated)...")
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
                                status_placeholder.info("Analyzing sentiment (aggregated)...")
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
                                status_placeholder.info("Running Gemini analysis (aggregated)...")
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
            mode_run = st.session_state.current_analysis_mode
            analysis_type_run = selected_analysis_type_key

            # ================= Display Compare Results =================
            if mode_run == "Compare Channels":
                if st.session_state.all_channel_results:
                    st.header("📊 Channel Comparison")
                    
                    # Get channel data
                    channel_data = []
                    for channel_id, results in st.session_state.all_channel_results.items():
                        channel_title = st.session_state.channel_info.get(channel_id, {}).get('title', channel_id)
                        metrics = results.get('total_metrics', {})
                        averages = results.get('averages', {})
                        channel_data.append({
                            'title': channel_title,
                            'id': channel_id,
                            'metrics': metrics,
                            'averages': averages,
                            'results': results
                        })

                    # Display side-by-side metrics comparison
                    st.subheader("📈 Performance Comparison")
                    
                    # Create columns for each channel
                    metric_cols = st.columns(len(channel_data))
                    
                    # Display total metrics for each channel
                    for idx, channel in enumerate(channel_data):
                        with metric_cols[idx]:
                            st.markdown(f"### {channel['title']}")
                            st.metric("Total Views", format_value("Views", channel['metrics'].get('views', 0)))
                            st.metric("Total Likes", format_value("Likes", channel['metrics'].get('likes', 0)))
                            st.metric("Total Comments", format_value("Comments", channel['metrics'].get('comments', 0)))
                    
                    # Create comparative visualizations
                    st.subheader("📊 Comparative Analysis")
                    
                    # Prepare data for comparison charts
                    comparison_data = {
                        'Channel': [],
                        'Views': [],
                        'Likes': [],
                        'Comments': [],
                        'Engagement Rate': []
                    }
                    
                    for channel in channel_data:
                        comparison_data['Channel'].append(channel['title'])
                        comparison_data['Views'].append(channel['metrics'].get('views', 0))
                        comparison_data['Likes'].append(channel['metrics'].get('likes', 0))
                        comparison_data['Comments'].append(channel['metrics'].get('comments', 0))
                        # Calculate engagement rate (likes + comments) / views * 100
                        views = channel['metrics'].get('views', 0)
                        engagement = (channel['metrics'].get('likes', 0) + channel['metrics'].get('comments', 0))
                        engagement_rate = (engagement / views * 100) if views > 0 else 0
                        comparison_data['Engagement Rate'].append(engagement_rate)
                    
                    # Create comparison DataFrame
                    df_comparison = pd.DataFrame(comparison_data)
                    
                    # Create metrics comparison charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Views and Engagement Bar Chart
                        fig_views = px.bar(df_comparison, 
                                         x='Channel', 
                                         y=['Views', 'Likes', 'Comments'],
                                         title='Performance Metrics Comparison',
                                         barmode='group')
                        st.plotly_chart(fig_views, use_container_width=True)
                    
                    with col2:
                        # Engagement Rate Comparison
                        fig_engagement = px.bar(df_comparison,
                                              x='Channel',
                                              y='Engagement Rate',
                                              title='Engagement Rate Comparison (%)',
                                              color='Channel')
                        st.plotly_chart(fig_engagement, use_container_width=True)
                    
                    # Video Performance Analysis
                    st.subheader("📺 Video Performance Analysis")
                    
                    # Create a combined video performance dataframe
                    video_data = []
                    for channel in channel_data:
                        channel_df = channel['results'].get('dataframe', pd.DataFrame())
                        if not channel_df.empty:
                            channel_df['Channel'] = channel['title']
                            video_data.append(channel_df)
                    
                    if video_data:
                        combined_df = pd.concat(video_data)
                        
                        # Create scatter plot of views vs engagement
                        fig_scatter = px.scatter(combined_df,
                                               x='Views',
                                               y='Likes',
                                               color='Channel',
                                               size='Comments',
                                               hover_data=['Title', 'Published'],
                                               title='Video Performance: Views vs Likes')
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Show video performance over time
                        fig_timeline = px.line(combined_df,
                                             x='Published',
                                             y='Views',
                                             color='Channel',
                                             title='Views Over Time')
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Sentiment Analysis Comparison
                    if analysis_type_run != "Channel Summary":
                        st.subheader("💭 Sentiment Analysis Comparison")
                        
                        # Compare sentiment patterns between channels
                        for channel in channel_data:
                            st.markdown(f"#### {channel['title']}")
                            
                            # Get sentiment details and video sentiments
                            sentiment_details = channel['results'].get('sentiment_details', {})
                            video_sentiments = channel['results'].get('video_sentiments', [])
                            
                            # Display confidence scores if available
                            avg_confidence = sentiment_details.get('average_confidence', {})
                            if avg_confidence:
                                st.markdown("**Average Sentiment Confidence:**")
                                st.markdown(f"""
                                - Positive: {avg_confidence.get('POSITIVE', 0):.2%}
                                - Neutral: {avg_confidence.get('NEUTRAL', 0):.2%}
                                - Negative: {avg_confidence.get('NEGATIVE', 0):.2%}
                                """)
                            
                            # Display top keywords if available
                            top_keywords = sentiment_details.get('top_keywords', [])
                            if top_keywords:
                                st.markdown("**Top Keywords:**")
                                keyword_text = ", ".join([f"`{kw[0]}`" for kw in top_keywords[:5]])
                                st.markdown(keyword_text)
                            
                            # Display Gemini's enhanced analysis if available
                            if 'overall_enhanced_analysis' in channel['results']:
                                with st.expander("🧠 View Detailed Analysis"):
                                    st.markdown(channel['results']['overall_enhanced_analysis'])
                            
                            # Display sample comments with sentiment
                            if video_sentiments:
                                with st.expander("💬 View Sample Comments"):
                                    for video in video_sentiments[:2]:  # Show comments from top 2 videos
                                        if 'comments' in video:
                                            st.markdown(f"**{video['title'][:50]}...**")
                                            for comment in video['comments'][:2]:  # Show top 2 comments per video
                                                sentiment_color = "#00CC96" if comment['sentiment'] == 'POSITIVE' else (
                                                    "#636EFA" if comment['sentiment'] == 'NEUTRAL' else "#EF553B")
                                                st.markdown(f"""
                                                <div style="border-left: 4px solid {sentiment_color}; padding-left: 10px; margin-bottom: 10px;">
                                                    <p style="font-style: italic; font-size: 14px;">{comment['text'][:150]}...</p>
                                                    <p style="font-size: 12px; color: gray;">Sentiment: {comment['sentiment']} (Score: {comment.get('score', 0):.2f})</p>
                                                </div>
                                                """, unsafe_allow_html=True)
                            
                            st.divider()
                        
                        # Add comparative insights
                        if len(channel_data) > 1:
                            st.markdown("### 📊 Comparative Insights")
                            try:
                                # Create a comparative analysis prompt for Gemini
                                comparative_prompt = f"""Compare the sentiment analysis results between these YouTube channels:

Channel Data:
{chr(10).join(f'- {ch["title"]}: Positive {ch["results"].get("sentiment_analysis", {}).get("POSITIVE", 0)}, ' + 
              f'Neutral {ch["results"].get("sentiment_analysis", {}).get("NEUTRAL", 0)}, ' + 
              f'Negative {ch["results"].get("sentiment_analysis", {}).get("NEGATIVE", 0)}' 
              for ch in channel_data)}

Please provide:
1. Key differences in audience sentiment
2. Comparative engagement patterns
3. Recommendations for each channel
4. Best practices identified

Format the response in clear sections with bullet points."""

                                if model is not None:
                                    response = model.generate_content(
                                        comparative_prompt,
                                        generation_config={
                                            'temperature': 0.7,
                                            'candidate_count': 1,
                                        }
                                    )
                                    if response.text:
                                        st.markdown(response.text)
                            except Exception as e:
                                logger.error(f"Error generating comparative insights: {e}")
                                st.warning("Unable to generate comparative insights at this time.")
                    
                    # Gemini Analysis Comparison
                    if analysis_type_run != "Comment Analysis":
                        st.subheader("🧠 Channel Analysis Comparison")
                        
                        analysis_cols = st.columns(len(channel_data))
                        for idx, channel in enumerate(channel_data):
                            with analysis_cols[idx]:
                                st.markdown(f"### {channel['title']}")
                                st.markdown(channel['results'].get('gemini_analysis', "No analysis available."))
                    
                    st.divider()
                    st.success(f"✅ Comparison Analysis ({analysis_type_run}) completed!")

                    # Add download button for comparison analysis
                    add_download_button(
                        st.session_state.all_channel_results,
                        analysis_type_run,
                        st.session_state.channel_info
                    )
                else:
                    st.warning("Comparison analysis ran, but no results.")

            # ================= Display Aggregate Results =================
            elif mode_run == "Aggregate Analysis":
                if st.session_state.aggregate_results:
                    agg_results = st.session_state.aggregate_results
                    agg_df = agg_results.get('dataframe', pd.DataFrame())
                    agg_figures = st.session_state.aggregate_figures
                    st.header("📊 Performance Overview (Aggregated)")

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
                    st.subheader("📈 Performance Trends (Aggregated)")

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
                                                agg_figures, "Aggregated")

                    if analysis_type_run != "Comment Analysis":
                        st.divider()
                        st.header("🧠 Gemini Analysis (Aggregated)")
                        st.markdown(agg_results.get('gemini_analysis', "No analysis available."))

                    st.divider()
                    st.success(f"✅ Aggregate Analysis ({analysis_type_run}) completed!")

                    # Add download button for aggregate analysis
                    add_download_button(
                        st.session_state.aggregate_results,
                        analysis_type_run
                    )
                else:
                    st.warning("Aggregate analysis ran, but no results.")
            else:
                st.markdown("### No Analysis Results Yet\nConfigure settings and click **\"Analyze\"**.")

# --- Entry Point ---
if __name__ == "__main__":
    if sentiment_pipeline is None:
        st.warning("Local sentiment model failed. Comment analysis unavailable.", icon="⚠️")
    main()