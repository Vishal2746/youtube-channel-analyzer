# youtube_analyzer.py (Corrected - No st calls)

import time
import random
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import backoff
from google.api_core import exceptions as google_exceptions
from transformers import pipeline
import json
from pathlib import Path
from itertools import cycle
from cache_manager import analysis_cache

# --- Logging setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Custom Exception ---
class APIQuotaExceeded(Exception):
    """Custom exception for API quota exceeded"""
    pass

# --- YouTube API Backoff ---
youtube_backoff_settings = {
    'wait_gen': backoff.expo,
    'exception': (HttpError, google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable),
    'max_tries': 3, 'max_time': 60,
    'giveup': lambda e: isinstance(e, HttpError) and e.resp.status == 403
}

def youtube_api_backoff(func):
    @backoff.on_exception(**youtube_backoff_settings)
    def wrapper(*args, **kwargs):
        try: return func(*args, **kwargs)
        except HttpError as e:
            logger.warning(f"YT API HttpError in {func.__name__}: {e.resp.status}. Retrying...")
            if e.resp.status in [403, 429]: raise APIQuotaExceeded(f"YT API quota/permission error in {func.__name__} ({e.resp.status})") from e
            raise
        except (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable) as e:
             logger.warning(f"YT API Resource/Service Error in {func.__name__}: {type(e).__name__}. Retrying...")
             raise
    return wrapper

# --- YouTube API Functions ---

class EnhancedChannelCache:
    """Enhanced cache system for YouTube data with tiered caching"""
    def __init__(self):
        self.cache_dir = "cache"
        self.channel_cache_file = os.path.join(self.cache_dir, "channel_cache.json")
        self.video_cache_file = os.path.join(self.cache_dir, "video_cache.json")
        self.comment_cache_file = os.path.join(self.cache_dir, "comment_cache.json")
        self._ensure_cache_dir()
        self.cache_durations = {
            'channel': 7 * 24 * 3600,  # 7 days for channel info
            'videos': 24 * 3600,      # 24 hours for videos
            'comments': 12 * 3600     # 12 hours for comments
        }
        self.channel_cache = self._load_cache(self.channel_cache_file)
        self.video_cache = self._load_cache(self.video_cache_file)
        self.comment_cache = self._load_cache(self.comment_cache_file)

    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _load_cache(self, cache_file):
        """Load cache from file"""
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading cache {cache_file}: {e}")
            return {}

    def _save_cache(self, cache_data, cache_file):
        """Save cache to file"""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.error(f"Error saving cache {cache_file}: {e}")

    def get_channel_data(self, channel_id):
        """Get channel data from cache if not expired"""
        cache_entry = self.channel_cache.get(channel_id, {})
        if cache_entry and time.time() - cache_entry.get('timestamp', 0) < self.cache_durations['channel']:
            return cache_entry.get('data')
        return None

    def get_video_data(self, video_id):
        """Get video data from cache if not expired"""
        cache_entry = self.video_cache.get(video_id, {})
        if cache_entry and time.time() - cache_entry.get('timestamp', 0) < self.cache_durations['videos']:
            return cache_entry.get('data')
        return None

    def get_comments_data(self, video_id):
        """Get comments data from cache if not expired"""
        cache_entry = self.comment_cache.get(video_id, {})
        if cache_entry and time.time() - cache_entry.get('timestamp', 0) < self.cache_durations['comments']:
            return cache_entry.get('data')
        return None

    def cache_channel_data(self, channel_id, data):
        """Cache channel data with timestamp"""
        self.channel_cache[channel_id] = {
            'data': data,
            'timestamp': time.time()
        }
        self._save_cache(self.channel_cache, self.channel_cache_file)

    def cache_video_data(self, video_id, data):
        """Cache video data with timestamp"""
        self.video_cache[video_id] = {
            'data': data,
            'timestamp': time.time()
        }
        self._save_cache(self.video_cache, self.video_cache_file)

    def cache_comments_data(self, video_id, data):
        """Cache comments data with timestamp"""
        self.comment_cache[video_id] = {
            'data': data,
            'timestamp': time.time()
        }
        self._save_cache(self.comment_cache, self.comment_cache_file)

    def clear_expired_cache(self):
        """Clear expired entries from all caches"""
        current_time = time.time()
        
        # Clear expired channel cache
        self.channel_cache = {
            k: v for k, v in self.channel_cache.items()
            if current_time - v.get('timestamp', 0) < self.cache_durations['channel']
        }
        
        # Clear expired video cache
        self.video_cache = {
            k: v for k, v in self.video_cache.items()
            if current_time - v.get('timestamp', 0) < self.cache_durations['videos']
        }
        
        # Clear expired comment cache
        self.comment_cache = {
            k: v for k, v in self.comment_cache.items()
            if current_time - v.get('timestamp', 0) < self.cache_durations['comments']
        }
        
        # Save cleaned caches
        self._save_cache(self.channel_cache, self.channel_cache_file)
        self._save_cache(self.video_cache, self.video_cache_file)
        self._save_cache(self.comment_cache, self.comment_cache_file)

# Initialize enhanced cache
enhanced_cache = EnhancedChannelCache()

class ChannelCache:
    """Cache system for YouTube channel IDs"""
    def __init__(self):
        self.cache_file = "channel_cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load channel IDs from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading channel cache: {e}")
            return {}
    
    def _save_cache(self):
        """Save channel IDs to cache file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Error saving channel cache: {e}")
    
    def get_channel_id(self, url):
        """Get channel ID from cache"""
        return self.cache.get(url)
    
    def add_channel_id(self, url, channel_id):
        """Add channel ID to cache"""
        self.cache[url] = channel_id
        self._save_cache()

class ChannelDataCache:
    """Cache system for YouTube channel data"""
    def __init__(self):
        self.cache_file = "channel_data_cache.json"
        self.cache = self._load_cache()
        self.cache_duration = 3600  # Cache duration in seconds (1 hour)
    
    def _load_cache(self):
        """Load channel data from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading channel data cache: {e}")
            return {}
    
    def _save_cache(self):
        """Save channel data to cache file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Error saving channel data cache: {e}")
    
    def get_channel_data(self, channel_id):
        """Get channel data from cache if not expired"""
        if channel_id in self.cache:
            cache_entry = self.cache[channel_id]
            cache_time = cache_entry.get('timestamp', 0)
            if time.time() - cache_time < self.cache_duration:
                return cache_entry.get('data')
        return None
    
    def add_channel_data(self, channel_id, data):
        """Add channel data to cache with timestamp"""
        self.cache[channel_id] = {
            'data': data,
            'timestamp': time.time()
        }
        self._save_cache()

# Initialize caches
channel_cache = ChannelCache()
channel_data_cache = ChannelDataCache()

class YouTubeAPIKeyManager:
    """Manages multiple YouTube API keys with rotation"""
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.key_cycle = cycle(self.api_keys) if self.api_keys else None
        self.current_key = next(self.key_cycle) if self.key_cycle else None

    def _load_api_keys(self):
        """Load API keys from environment variables"""
        keys = []
        # Load primary key
        primary_key = os.getenv("YOUTUBE_API_KEY")
        if primary_key:
            keys.append(primary_key)
        
        # Load additional keys (YOUTUBE_API_KEY_1, YOUTUBE_API_KEY_2, etc.)
        i = 1
        while True:
            key = os.getenv(f"YOUTUBE_API_KEY_{i}")
            if not key:
                break
            keys.append(key)
            i += 1
        
        return keys

    def get_next_key(self):
        """Get the next API key in rotation"""
        if not self.key_cycle:
            return None
        self.current_key = next(self.key_cycle)
        return self.current_key

    def get_current_key(self):
        """Get the current API key"""
        return self.current_key

# Initialize the API key manager
api_key_manager = YouTubeAPIKeyManager()

@youtube_api_backoff
def get_channel_id(youtube, url):
    """Extract channel ID from various YouTube URL formats with backoff."""
    logger.debug(f"Attempting to get channel ID for URL: {url}")
    
    # Check cache first
    cached_id = channel_cache.get_channel_id(url)
    if cached_id:
        logger.debug(f"Found channel ID in cache for URL: {url}")
        return cached_id
    
    try:
        channel_id = None
        if '/@' in url:
            handle = url.split('/@')[1].split('/')[0]
            logger.debug(f"Detected handle: {handle}")
            response = youtube.search().list(part='id', q=f'@{handle}', type='channel', maxResults=1).execute()
            if response.get('items'): 
                channel_id = response['items'][0]['id']['channelId']
        elif '/channel/' in url:
            channel_id = url.split('/channel/')[1].split('/')[0]
        elif '/user/' in url:
            username = url.split('/user/')[1].split('/')[0]
            logger.debug(f"Detected username: {username}")
            response = youtube.channels().list(part='id', forUsername=username).execute()
            if response.get('items'): 
                channel_id = response['items'][0]['id']
        else:
            logger.warning(f"Unsupported URL format, trying fallback search: {url}")
            possible_name = url.split('/')[-1]
            if possible_name and possible_name not in ['featured', 'videos', 'playlists', 'community', 'about']:
                response = youtube.search().list(part='id', q=possible_name, type='channel', maxResults=1).execute()
                if response.get('items'):
                    channel_id = response['items'][0]['id']['channelId']
                    logger.warning(f"Found channel ID via fallback search (might be inaccurate): {channel_id}")
        
        if channel_id:
            # Store in cache if found
            channel_cache.add_channel_id(url, channel_id)
            return channel_id
        else:
            logger.warning(f"Could not determine channel ID for: {url}")
            return None
            
    except APIQuotaExceeded: 
        # Try with next API key
        next_key = api_key_manager.get_next_key()
        if next_key:
            logger.info("Switching to next API key due to quota exceeded")
            youtube = build('youtube', 'v3', developerKey=next_key)
            return get_channel_id(youtube, url)  # Retry with new key
        raise  # If no more keys available, raise the exception
    except Exception as e:
        logger.error(f"Error getting channel ID for {url}: {e}", exc_info=True)
        return None

# **** ENSURE THIS FUNCTION IS PRESENT ****
@youtube_api_backoff
def get_video_comments_page(youtube, video_id, max_results_page=100, page_token=None):
     """Fetches a single page of comments for a video."""
     logger.debug(f"Fetching comments page for video {video_id}, pageToken: {page_token}")
     request = youtube.commentThreads().list(
          part='snippet',
          videoId=video_id,
          maxResults=max_results_page,
          order='relevance',
          textFormat='plainText',
          pageToken=page_token
     )
     response = request.execute()
     return response

def get_youtube_videos_grouped(channel_urls, max_results_per_channel=10, status_placeholder=None):
    """Get videos from YouTube channels with caching"""
    if not api_key_manager.get_current_key():
        logger.error("Critical Error: No YouTube API keys configured.")
        raise ValueError("YouTube API key is not configured.")

    videos_by_channel = defaultdict(list)
    channel_info = {}
    youtube = None

    try:
        youtube = build('youtube', 'v3', developerKey=api_key_manager.get_current_key())
    except Exception as e:
        logger.error(f"Error initializing YouTube client: {e}")
        raise ConnectionError(f"Fatal Error: Could not initialize YouTube client: {str(e)}") from e

    def update_status(message):
        if status_placeholder:
            status_placeholder.info(message)
        logger.info(message)

    processed_channel_ids = set()

    for i, url in enumerate(channel_urls):
        current_url_short = url.split('/')[-1] or url.split('/')[-2]
        update_status(f"Processing channel {i+1}/{len(channel_urls)}: ...{current_url_short}")

        try:
            channel_id = get_channel_id(youtube, url)
            if not channel_id:
                continue

            # Check if we have cached analysis for this channel
            cached_analysis = analysis_cache.get_analysis(channel_id, 'channel_analysis')
            if cached_analysis:
                logger.info(f"Using cached analysis for channel {channel_id}")
                videos_by_channel[channel_id] = cached_analysis.get('videos', [])
                channel_info[channel_id] = cached_analysis.get('info', {})
                processed_channel_ids.add(channel_id)
                continue

            # Check cache for channel data
            cached_data = enhanced_cache.get_channel_data(channel_id)
            if cached_data:
                logger.info(f"Using cached data for channel: {channel_id}")
                videos_by_channel[channel_id] = cached_data.get('videos', [])
                channel_info[channel_id] = cached_data.get('info', {})
                processed_channel_ids.add(channel_id)
                continue

            # Get channel title and uploads playlist ID
            channel_response = youtube.channels().list(
                part='contentDetails,snippet', id=channel_id
            ).execute()

            if not channel_response.get('items'):
                logger.warning(f"No channel data found for ID: {channel_id} ({url}). Skipping.")
                continue

            channel_data = channel_response['items'][0]
            channel_title = channel_data.get('snippet', {}).get('title', channel_id)
            uploads_playlist_id = channel_data.get('contentDetails', {}).get('relatedPlaylists', {}).get('uploads')

            if not uploads_playlist_id:
                logger.warning(f"Could not find uploads playlist for channel: {channel_title} ({channel_id}). Skipping videos.")
                channel_info[channel_id] = {'title': channel_title, 'url': url}
                processed_channel_ids.add(channel_id)
                continue

            channel_info[channel_id] = {'title': channel_title, 'url': url}
            processed_channel_ids.add(channel_id)
            update_status(f"Fetching videos for: {channel_title}...")

            # Fetch videos and store in cache
            videos = []
            next_page_token = None
            videos_fetched_count = 0

            while videos_fetched_count < max_results_per_channel:
                results_to_fetch = min(50, max_results_per_channel - videos_fetched_count)
                if results_to_fetch <= 0:
                    break

                try:
                    pl_response = youtube.playlistItems().list(
                        part='snippet', playlistId=uploads_playlist_id,
                        maxResults=results_to_fetch, pageToken=next_page_token
                    ).execute()
                    
                    found = pl_response.get('items', [])
                    video_ids = [item['snippet']['resourceId']['videoId'] for item in found 
                               if item.get('snippet', {}).get('resourceId', {}).get('kind') == 'youtube#video']
                    
                    if video_ids:
                        vid_response = youtube.videos().list(
                            part='snippet,statistics,contentDetails', id=','.join(video_ids)
                        ).execute()
                        fetched_videos = vid_response.get('items', [])
                        videos.extend(fetched_videos)
                        videos_fetched_count += len(fetched_videos)
                    
                    next_page_token = pl_response.get('nextPageToken')
                    if not next_page_token:
                        break
                    time.sleep(0.1)
                    
                except APIQuotaExceeded:
                    raise
                except Exception as e:
                    logger.error(f"Error fetching videos for {channel_title}: {e}")
                    break

            videos_by_channel[channel_id] = videos
            
            # Cache the channel data
            enhanced_cache.cache_channel_data(channel_id, {
                'videos': videos,
                'info': channel_info[channel_id]
            })

            # After fetching and processing videos, cache the complete analysis
            analysis_data = {
                'videos': videos_by_channel[channel_id],
                'info': channel_info[channel_id],
                'timestamp': datetime.now().isoformat()
            }
            analysis_cache.add_analysis(channel_id, 'channel_analysis', analysis_data)

        except APIQuotaExceeded as e:
            # Try with next API key
            next_key = api_key_manager.get_next_key()
            if next_key:
                logger.info("Switching to next API key due to quota exceeded")
                youtube = build('youtube', 'v3', developerKey=next_key)
                continue  # Retry this channel with new key
            logger.error(f"All API keys exhausted: {e}")
            raise
        except HttpError as e:
            logger.error(f"HTTP Error processing channel {url}: {e}")
            # Log but continue to the next channel
        except Exception as e:
            logger.error(f"Unexpected error processing channel {url}: {e}", exc_info=True)
            # Log but continue to the next channel

    # Clear status only if placeholder was provided
    if status_placeholder: status_placeholder.empty()
    logger.info(f"Finished fetching. Found videos for {len(videos_by_channel)} channels.")
    return videos_by_channel, channel_info


def filter_videos_by_date_grouped(videos_by_channel, start_date_str, end_date_str):
    """Filters videos within each channel's list based on date."""
    # ... (Function content remains the same as previous version) ...
    filtered_groups = {}
    if not videos_by_channel: return {}
    try: start_dt = datetime.strptime(start_date_str, '%Y-%m-%d'); end_dt = datetime.strptime(end_date_str, '%Y-%m-%d') + timedelta(days=1)
    except ValueError: logger.error("Invalid date format for filtering."); return videos_by_channel
    for channel_id, video_list in videos_by_channel.items():
        filtered_list = []
        for video in video_list:
            published_at_str = video.get('snippet', {}).get('publishedAt');
            if not published_at_str: continue
            try: pub_dt = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'));
            except ValueError: continue
            if start_dt <= pub_dt.replace(tzinfo=None) < end_dt: filtered_list.append(video)
        if filtered_list: filtered_groups[channel_id] = filtered_list; logger.info(f"Channel {channel_id}: {len(filtered_list)} videos remain.")
        else: logger.info(f"Channel {channel_id}: No videos in date range.")
    return filtered_groups


# --- Hugging Face Sentiment Analysis ---
# ... (sentiment_pipeline loading, analyze_comment_sentiment_hf remain the same) ...
sentiment_pipeline = None
try:
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"; logger.info(f"Loading HF model: {model_name}...")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name); logger.info("HF model loaded.")
except Exception as e: logger.error(f"CRITICAL: Failed load HF model ('{model_name}'). Sentiment disabled. Error: {e}", exc_info=True)

def analyze_comment_sentiment_hf(comment_text):
    if sentiment_pipeline is None: return {"sentiment": "NEUTRAL", "score": 0.0, "details": {}}
    if not comment_text or not isinstance(comment_text, str) or len(comment_text.strip()) == 0: 
        return {"sentiment": "NEUTRAL", "score": 0.0, "details": {}}
    try:
        truncated_text = comment_text[:500]
        results = sentiment_pipeline(truncated_text)
        if results:
            result = results[0]
            sentiment_label = result['label'].upper()
            confidence = result['score']
            
            # Map the sentiment labels
            if sentiment_label == 'LABEL_0': 
                sentiment = 'NEGATIVE'
            elif sentiment_label == 'LABEL_1': 
                sentiment = 'NEUTRAL'
            elif sentiment_label == 'LABEL_2': 
                sentiment = 'POSITIVE'
            elif sentiment_label in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']: 
                sentiment = sentiment_label
            else: 
                logger.warning(f"Unexpected HF label '{result['label']}'. Default NEUTRAL.")
                sentiment = 'NEUTRAL'
                
            # Extract key phrases and topics (simple implementation)
            words = comment_text.lower().split()
            # Basic keyword extraction (could be enhanced with NLP techniques)
            keywords = [word for word in words if len(word) > 4 and word.isalpha()][:5]
            
            return {
                "sentiment": sentiment,
                "score": confidence,
                "details": {
                    "original_label": result['label'],
                    "confidence": confidence,
                    "keywords": keywords,
                    "text_sample": truncated_text[:100] + "..." if len(truncated_text) > 100 else truncated_text
                }
            }
        else: 
            logger.warning(f"HF pipeline no result: '{truncated_text[:50]}...'")
            return {"sentiment": "NEUTRAL", "score": 0.0, "details": {}}
    except Exception as e: 
        logger.error(f"Error HF analysis: '{comment_text[:50]}...': {e}")
        return {"sentiment": "NEUTRAL", "score": 0.0, "details": {}}

def get_video_comments_with_sentiment(youtube, video_id, max_total_comments=20):
    """Gets comments for a video with enhanced caching and detailed sentiment analysis"""
    # Try to get from cache first
    cached_comments = enhanced_cache.get_comments_data(video_id)
    if cached_comments:
        logger.info(f"Using cached comments for video {video_id}")
        return cached_comments

    comments_data = []
    analyzed_count = 0
    page_token = None
    quota_exceeded_youtube = False
    sentiment_analysis_enabled = sentiment_pipeline is not None

    if not sentiment_analysis_enabled:
        logger.warning(f"Sentiment analysis disabled for video {video_id} as HF model is not loaded.")

    while analyzed_count < max_total_comments and not quota_exceeded_youtube:
        try:
            results_on_page = min(100, max_total_comments - analyzed_count)
            if results_on_page <= 0: break

            # **** ENSURE THIS CALL IS CORRECT ****
            response = get_video_comments_page(youtube, video_id, results_on_page, page_token)
            # ****                        ****

            items = response.get('items', [])
            if not items: break

            for item in items:
                if analyzed_count >= max_total_comments: break
                try:
                    top_comment = item.get('snippet', {}).get('topLevelComment', {}).get('snippet', {})
                    comment_text = top_comment.get('textDisplay', '')
                    if not comment_text: continue

                    sentiment_data = {"sentiment": "NEUTRAL", "score": 0.0, "details": {}}
                    if sentiment_analysis_enabled:
                        sentiment_data = analyze_comment_sentiment_hf(comment_text)

                    comments_data.append({
                        'text': comment_text, 
                        'sentiment': sentiment_data["sentiment"], 
                        'sentiment_score': sentiment_data["score"],
                        'sentiment_details': sentiment_data["details"],
                        'likes': top_comment.get('likeCount', 0), 
                        'published_at': top_comment.get('publishedAt', '')
                    })
                    analyzed_count += 1
                except Exception as item_err:
                    logger.error(f"Error processing comment item for video {video_id}: {item_err}", exc_info=True)
                    continue

            if analyzed_count >= max_total_comments: break
            page_token = response.get('nextPageToken')
            if not page_token: break
            time.sleep(0.1)

        except APIQuotaExceeded as e:
            logger.error(f"YouTube API Quota Exceeded while fetching comments for video {video_id}: {e}")
            quota_exceeded_youtube = True # Stop fetching comments for this video
            # Raise the exception so app.py can handle UI
            raise
        except Exception as page_err:
             logger.error(f"Unexpected error fetching comment page for video {video_id}: {page_err}", exc_info=True)
             # Decide whether to break or raise? Let's raise for app.py to handle.
             raise RuntimeError(f"Failed to fetch comment page for {video_id}") from page_err

    if quota_exceeded_youtube:
        logger.warning(f"YouTube quota likely exceeded during comment fetch. Returning partial data for {video_id}.")
        # Don't raise here if we got *some* data before quota hit

    logger.info(f"Retrieved and analyzed {len(comments_data)} comments for video {video_id}.")
    
    # Cache the results before returning
    if comments_data:
        enhanced_cache.cache_comments_data(video_id, comments_data)
    
    return comments_data

# --- analyze_channel_performance ---
def analyze_channel_performance(videos, max_comments=20):
    if not videos: logger.warning("No videos provided."); return None
    analysis = {
        'total_metrics': {'views': 0, 'likes': 0, 'comments': 0}, 
        'averages': {'views': 0, 'likes': 0, 'comments': 0}, 
        'sentiment_analysis': {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}, 
        'sentiment_details': {
            'average_confidence': {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0},
            'top_keywords': {},
            'sentiment_distribution': []
        },
        'video_sentiments': []
    }
    num_videos = len(videos)
    try: # Basic metrics
        total_views = sum(int(v['statistics'].get('viewCount', 0)) for v in videos)
        total_likes = sum(int(v['statistics'].get('likeCount', 0)) for v in videos)
        total_comments_stat = sum(int(v['statistics'].get('commentCount', 0)) for v in videos)
        analysis['total_metrics'] = {'views': total_views, 'likes': total_likes, 'comments': total_comments_stat}
        if num_videos > 0: analysis['averages'] = {'views': total_views / num_videos, 'likes': total_likes / num_videos, 'comments': total_comments_stat / num_videos}
    except Exception as e: logger.error(f"Error calculating basic metrics: {e}")

    youtube = None; youtube_client_initialized = False
    try: # Sentiment Analysis
        if max_comments > 0 and os.getenv("YOUTUBE_API_KEY"):
            try: youtube = build('youtube', 'v3', developerKey=os.getenv("YOUTUBE_API_KEY")); youtube_client_initialized = True
            except Exception as build_err: logger.error(f"Failed to build YT client: {build_err}")
        
        # Initialize Gemini for enhanced analysis
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemini_available = False
        gemini_model = None
        
        if gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
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
                gemini_model = genai.GenerativeModel(
                    model_name='gemini-2.0-flash',
                    safety_settings=safety_settings
                )
                gemini_available = True
                logger.info("Gemini initialized for enhanced sentiment analysis")
            except Exception as e:
                logger.error(f"Error initializing Gemini: {e}")
                gemini_available = False
        
        # Track sentiment confidence sums and counts for averaging
        sentiment_confidence_sums = {'POSITIVE': 0.0, 'NEGATIVE': 0.0, 'NEUTRAL': 0.0}
        sentiment_confidence_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
        all_keywords = []
        all_comments_for_analysis = []
        
        for video in videos:
            video_id = video.get('id', 'N/A')
            video_title = video.get('snippet', {}).get('title', 'Unknown Title')
            video_sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
            video_sentiment_scores = {'POSITIVE': [], 'NEGATIVE': [], 'NEUTRAL': []}
            video_keywords = []
            analyzed_comment_count = 0
            video_comments = []
            
            if max_comments > 0 and youtube:
                try:
                    logger.info(f"Fetching/Analyzing comments for: {video_title[:50]}... ({video_id})")
                    comments = get_video_comments_with_sentiment(youtube, video_id, max_total_comments=max_comments)
                    analyzed_comment_count = len(comments)
                    
                    if comments:
                        for comment in comments:
                            sentiment = comment.get('sentiment', 'NEUTRAL').upper()
                            score = comment.get('sentiment_score', 0.0)
                            details = comment.get('sentiment_details', {})
                            
                            # Store comment for Gemini analysis
                            video_comments.append({
                                'text': comment.get('text', ''),
                                'sentiment': sentiment,
                                'score': score
                            })
                            all_comments_for_analysis.append(comment.get('text', ''))
                            
                            # Count sentiments
                            if sentiment in video_sentiment_counts: 
                                video_sentiment_counts[sentiment] += 1
                                analysis['sentiment_analysis'][sentiment] += 1
                                
                                # Track scores for averaging
                                video_sentiment_scores[sentiment].append(score)
                                sentiment_confidence_sums[sentiment] += score
                                sentiment_confidence_counts[sentiment] += 1
                            else: 
                                video_sentiment_counts['NEUTRAL'] += 1
                                analysis['sentiment_analysis']['NEUTRAL'] += 1
                            
                            # Collect keywords
                            keywords = details.get('keywords', [])
                            video_keywords.extend(keywords)
                            all_keywords.extend(keywords)
                    else: 
                        logger.debug(f"No comments analyzed/returned for {video_id}.")
                except APIQuotaExceeded as e:
                    logger.error(f"Halting comment analysis due to YouTube Quota Error for video {video_id}: {e}")
                    raise APIQuotaExceeded(f"Quota hit during comment fetch for {video_id}") from e
                except Exception as e:
                    logger.error(f"Error in comment processing loop for video {video_id}: {e}", exc_info=True)
            
            # Calculate average sentiment scores for this video
            video_avg_scores = {}
            for sentiment, scores in video_sentiment_scores.items():
                if scores:
                    video_avg_scores[sentiment] = sum(scores) / len(scores)
                else:
                    video_avg_scores[sentiment] = 0.0
            
            # Get top keywords for this video
            keyword_counts = {}
            for keyword in video_keywords:
                if keyword in keyword_counts:
                    keyword_counts[keyword] += 1
                else:
                    keyword_counts[keyword] = 1
            
            top_video_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Get enhanced analysis from Gemini for this video's comments
            enhanced_analysis = None
            if gemini_available and video_comments:
                try:
                    # Create prompt for Gemini
                    prompt = f"""Analyze these YouTube comments for the video "{video_title}" and provide insights:

Comments with Sentiment:
{chr(10).join(f'- "{c["text"][:100]}..." (Sentiment: {c["sentiment"]}, Score: {c["score"]:.2f})' for c in video_comments[:5])}

Please provide:
1. Key themes and topics discussed
2. Overall sentiment patterns
3. Viewer engagement insights
4. Content feedback summary
5. Actionable recommendations

Format the response in clear sections with bullet points."""

                    response = gemini_model.generate_content(
                        prompt,
                        generation_config={
                            'temperature': 0.7,
                            'candidate_count': 1,
                        }
                    )
                    
                    if response.text:
                        enhanced_analysis = response.text
                        logger.info(f"Generated enhanced analysis for video: {video_title[:50]}...")
                except Exception as e:
                    logger.error(f"Error getting Gemini analysis for video {video_id}: {e}")
            
            # Store results even if comments failed/skipped
            actual_comments_on_video = int(video.get('statistics', {}).get('commentCount', 0))
            total_analyzed_for_report = sum(video_sentiment_counts.values())
            
            video_result = {
                'id': video_id, 
                'title': video_title, 
                'actual_comment_count': actual_comments_on_video, 
                'analyzed_comment_count': analyzed_comment_count, 
                'sentiments': video_sentiment_counts, 
                'sentiment_percentages': {
                    s: (c / total_analyzed_for_report * 100) if total_analyzed_for_report > 0 else 0 
                    for s, c in video_sentiment_counts.items()
                },
                'average_sentiment_scores': video_avg_scores,
                'top_keywords': top_video_keywords,
                'comments': video_comments[:5]  # Store sample comments
            }
            
            # Add enhanced analysis if available
            if enhanced_analysis:
                video_result['enhanced_analysis'] = enhanced_analysis
            
            analysis['video_sentiments'].append(video_result)
        
        # Get overall enhanced analysis from Gemini
        if gemini_available and all_comments_for_analysis:
            try:
                overall_prompt = f"""Analyze the overall sentiment patterns from these YouTube comments across multiple videos:

Sample Comments:
{chr(10).join(f'- "{comment[:100]}..."' for comment in all_comments_for_analysis[:10])}

Sentiment Distribution:
- Positive: {analysis['sentiment_analysis']['POSITIVE']}
- Neutral: {analysis['sentiment_analysis']['NEUTRAL']}
- Negative: {analysis['sentiment_analysis']['NEGATIVE']}

Please provide:
1. Overall audience sentiment analysis
2. Common themes and patterns
3. Content strategy recommendations
4. Engagement insights
5. Areas for improvement

Format the response in clear sections with bullet points."""

                overall_response = gemini_model.generate_content(
                    overall_prompt,
                    generation_config={
                        'temperature': 0.7,
                        'candidate_count': 1,
                    }
                )
                
                if overall_response.text:
                    analysis['overall_enhanced_analysis'] = overall_response.text
                    logger.info("Generated overall enhanced analysis")
            except Exception as e:
                logger.error(f"Error getting overall Gemini analysis: {e}")
        
        # Calculate overall average confidence scores
        for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            if sentiment_confidence_counts[sentiment] > 0:
                analysis['sentiment_details']['average_confidence'][sentiment] = (
                    sentiment_confidence_sums[sentiment] / sentiment_confidence_counts[sentiment]
                )
        
        # Get overall top keywords
        keyword_counts = {}
        for keyword in all_keywords:
            if keyword in keyword_counts:
                keyword_counts[keyword] += 1
            else:
                keyword_counts[keyword] = 1
        
        analysis['sentiment_details']['top_keywords'] = sorted(
            keyword_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        # Calculate sentiment distribution
        total_sentiments = sum(analysis['sentiment_analysis'].values())
        if total_sentiments > 0:
            analysis['sentiment_details']['sentiment_distribution'] = {
                sentiment: (count / total_sentiments * 100)
                for sentiment, count in analysis['sentiment_analysis'].items()
            }
        
        return analysis
    except APIQuotaExceeded:
        raise
    except Exception as e:
        logger.error(f"Fatal error in analyze_channel_performance: {e}", exc_info=True)
        raise RuntimeError("Error during channel performance analysis") from e