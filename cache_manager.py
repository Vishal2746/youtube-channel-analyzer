# cache_manager.py - Enhanced caching system for YouTube analysis

import os
import json
import logging
from datetime import datetime, timedelta
import hashlib

# Set up logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AnalysisCache:
    """Enhanced cache system for YouTube analysis results with persistent storage"""
    
    def __init__(self, cache_dir="analysis_cache"):
        """Initialize the cache with the specified directory"""
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
        self.cache_durations = {
            'channel_analysis': 7 * 24 * 3600,  # 7 days for channel analysis
            'sentiment_analysis': 3 * 24 * 3600,  # 3 days for sentiment analysis
            'comparison_analysis': 5 * 24 * 3600,  # 5 days for comparison analysis
            'ml_analysis': 5 * 24 * 3600,  # 5 days for ML analysis
        }
        logger.info(f"Analysis cache initialized at {self.cache_dir}")
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory: {self.cache_dir}")
    
    def _get_cache_file_path(self, identifier, analysis_type):
        """Get the file path for a cache entry"""
        # Create a hash of the identifier to use as filename
        if isinstance(identifier, list):
            # For multiple channels (comparison), sort and join with comma
            identifier = ",".join(sorted(identifier))
        
        hash_id = hashlib.md5(identifier.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{analysis_type}_{hash_id}.json")
    
    def get_analysis(self, identifier, analysis_type):
        """Get analysis from cache if not expired"""
        cache_file = self._get_cache_file_path(identifier, analysis_type)
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # Check if cache is expired
                timestamp = cache_data.get('timestamp')
                if timestamp:
                    cache_time = datetime.fromisoformat(timestamp)
                    cache_age = (datetime.now() - cache_time).total_seconds()
                    
                    if cache_age < self.cache_durations.get(analysis_type, 24 * 3600):
                        logger.info(f"Using cached {analysis_type} for {identifier[:30]}...")
                        return cache_data
                    else:
                        logger.info(f"Cache expired for {analysis_type} ({identifier[:30]})")
                        return None
            return None
        except Exception as e:
            logger.error(f"Error reading cache for {analysis_type} ({identifier[:30]}): {e}")
            return None
    
    def add_analysis(self, identifier, analysis_type, data):
        """Add analysis to cache with timestamp"""
        if not data:
            logger.warning(f"Attempted to cache empty data for {analysis_type}")
            return False
        
        cache_file = self._get_cache_file_path(identifier, analysis_type)
        
        try:
            # Ensure data has a timestamp
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now().isoformat()
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Cached {analysis_type} for {identifier[:30]}")
            return True
        except Exception as e:
            logger.error(f"Error caching {analysis_type} for {identifier[:30]}: {e}")
            return False
    
    def clear_expired_cache(self):
        """Clear expired entries from cache"""
        try:
            count = 0
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue
                
                # Extract analysis type from filename
                parts = filename.split('_', 1)
                if len(parts) != 2:
                    continue
                
                analysis_type = parts[0]
                cache_file = os.path.join(self.cache_dir, filename)
                
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    # Check if cache is expired
                    timestamp = cache_data.get('timestamp')
                    if timestamp:
                        cache_time = datetime.fromisoformat(timestamp)
                        cache_age = (datetime.now() - cache_time).total_seconds()
                        
                        if cache_age >= self.cache_durations.get(analysis_type, 24 * 3600):
                            os.remove(cache_file)
                            count += 1
                except Exception as e:
                    logger.error(f"Error checking cache file {filename}: {e}")
            
            logger.info(f"Cleared {count} expired cache entries")
            return count
        except Exception as e:
            logger.error(f"Error clearing expired cache: {e}")
            return 0
    
    def get_cache_stats(self):
        """Get statistics about the cache"""
        try:
            stats = {
                'total_entries': 0,
                'size_bytes': 0,
                'types': {}
            }
            
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue
                
                # Extract analysis type from filename
                parts = filename.split('_', 1)
                if len(parts) != 2:
                    continue
                
                analysis_type = parts[0]
                cache_file = os.path.join(self.cache_dir, filename)
                
                # Update stats
                stats['total_entries'] += 1
                stats['size_bytes'] += os.path.getsize(cache_file)
                
                if analysis_type not in stats['types']:
                    stats['types'][analysis_type] = 0
                stats['types'][analysis_type] += 1
            
            return stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}

# Initialize the analysis cache
analysis_cache = AnalysisCache()