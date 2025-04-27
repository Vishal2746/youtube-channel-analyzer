# nltk_setup.py - Handles NLTK resource downloading and initialization

import logging
import os
import nltk
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_nltk_resources():
    """Download and set up required NLTK resources"""
    try:
        # Create a directory for NLTK data if it doesn't exist
        nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Set NLTK data path
        nltk.data.path.append(nltk_data_dir)
        
        # List of required NLTK resources
        required_resources = [
            ('punkt', 'tokenizers/punkt'),
            ('stopwords', 'corpora/stopwords')
        ]
        
        # Download required resources
        for resource, resource_path in required_resources:
            try:
                # Check if resource exists
                try:
                    nltk.data.find(resource_path)
                    logger.info(f"NLTK resource '{resource}' already exists")
                except LookupError:
                    # Download the resource
                    logger.info(f"Downloading NLTK resource: {resource}")
                    nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
                    logger.info(f"Successfully downloaded NLTK resource: {resource}")
            except Exception as e:
                logger.error(f"Error downloading NLTK resource '{resource}': {e}")
                
        # Verify resources are available
        for resource, resource_path in required_resources:
            try:
                nltk.data.find(resource_path)
                logger.info(f"Verified NLTK resource '{resource}' is available")
            except LookupError:
                logger.error(f"NLTK resource '{resource}' is still not available after download attempt")
                
        return True
    except Exception as e:
        logger.error(f"Error setting up NLTK resources: {e}")
        return False

# Initialize NLTK resources
def initialize_nltk():
    """Initialize NLTK resources and return status"""
    try:
        # Create a simple stopwords list as fallback
        basic_stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                         'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                         'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                         'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                         'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                         'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                         'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                         'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                         'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                         'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                         'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                         'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])
        
        # Try to download and set up NLTK resources
        success = setup_nltk_resources()
        
        # If successful, use NLTK stopwords, otherwise use basic stopwords
        if success:
            try:
                from nltk.corpus import stopwords
                STOPWORDS = set(stopwords.words('english'))
                logger.info("Using NLTK stopwords")
            except Exception as e:
                logger.warning(f"Error loading NLTK stopwords: {e}. Using basic stopwords.")
                STOPWORDS = basic_stopwords
        else:
            logger.warning("Using basic stopwords due to NLTK setup failure")
            STOPWORDS = basic_stopwords
            
        # Create a simple word tokenizer as fallback
        def basic_tokenize(text):
            """Basic tokenization function that splits on whitespace and punctuation"""
            if not text:
                return []
            # Remove punctuation and split on whitespace
            import re
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            return [word for word in text.split() if word]
        
        # Try to use NLTK tokenizer, otherwise use basic tokenizer
        try:
            from nltk.tokenize import word_tokenize
            tokenize_function = word_tokenize
            logger.info("Using NLTK word_tokenize")
        except Exception as e:
            logger.warning(f"Error loading NLTK tokenizer: {e}. Using basic tokenizer.")
            tokenize_function = basic_tokenize
            
        return {
            'success': True,
            'stopwords': STOPWORDS,
            'tokenize': tokenize_function
        }
    except Exception as e:
        logger.error(f"Error initializing NLTK: {e}")
        return {
            'success': False,
            'stopwords': basic_stopwords,
            'tokenize': basic_tokenize
        }

# Initialize NLTK when this module is imported
nltk_resources = initialize_nltk()