import re
import logging
from functools import lru_cache, partial
import time
import urllib.parse
from urllib.parse import urlparse, urljoin, urlunparse
from typing import List, Dict, Optional, Set, Any
from enum import Enum
from dataclasses import dataclass, field
import concurrent.futures
import multiprocessing
import psutil

import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# nltk importation
NLTK_AVAILABLE = False
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    # from nltk.metrics import edit_distance

    # Download necessary NLTK resources if not present
    @lru_cache(maxsize=1)
    def _ensure_nltk_resource(resource):
        try:
            nltk.data.find(resource)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)

    def ensure_nltk_resources(resources=('punkt', 'stopwords', 'wordnet')):
        """Ensures multiple NLTK resources are available."""
        for resource in resources:
            if resource == 'punkt':
                _ensure_nltk_resource('tokenizers/punkt')
            elif resource.startswith('corpora/'):
                _ensure_nltk_resource(resource)
            else:
                _ensure_nltk_resource(f'corpora/{resource}')

    ensure_nltk_resources()

    STOPWORDS = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    NLTK_AVAILABLE = True

except ImportError:
    logger.warning(
        "NLTK not available. Some advanced features will be disabled.")
    STOPWORDS = set()

DATEUTIL_AVAILABLE = False
try:
    from dateutil.parser import parse as dateutil_parse
    DATEUTIL_AVAILABLE = True
except ImportError:
    logger.warning("dateutil not available. Datetime features will be disabled.")

WORD_SPLITTER_AVAILABLE = False
try:
    from word_splitter import general_word_splitter
    WORD_SPLITTER_AVAILABLE = True
except ImportError:
    logger.warning(
        "Word splitter module not available. Some of the splitting functionalities wont be available ..."
    )

# Define constants
class TextCleanerConstants:
    # Regex patterns
    NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
    EXTRA_SPACE_RE = re.compile(r"\s+")
    SINGLE_CHAR_RE = re.compile(r"\b[A-Za-z]\b")
    SENTENCE_END_RE = re.compile(r"([.?!])\s+(?=[A-Z])")
    PUNCTUATION_SPACE_RE = re.compile(r"([.,?!])(?! )")
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
    NUMBER_PATTERN = re.compile(r'\b\d+\b')
    
    # URL options
    URL_OPTIONS = {
        'default_scheme': 'https',
        'normalize_case': True,
        'remove_fragments': True,
        'remove_query_params': [
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'fbclid', 'gclid', 'msclkid', 'zanpid', 'dclid', '_ga'
        ],
        'required_query_params': None,
        'remove_trailing_slash': True,
        'remove_www': True
    }

    # Datetime constants
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    DEFAULT_DATETIME_FORMAT = DATETIME_FORMAT
    DATETIME_PATTERNS = [
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',
        r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
        r'([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})',
        r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\s+\d{1,2}:\d{2}:\d{2})',
        r'([A-Za-z]+,\s+[A-Za-z]+\s+\d{1,2},\s+\d{4})'
    ]
    
    # Language support
    SUPPORTED_LANGUAGES = {'english', 'spanish', 'french', 'german', 'italian', 'portuguese', 'dutch'}


# defining the preprocessing strategy
class ProcessingStrategy(Enum):
    SEQUENTIAL = 'sequential'
    MULTIPROCESSING = 'multiprocessing'
    VECTORIZE = 'vectorize'
    AUTO = 'auto'

@dataclass
class CleanerConfig:
    """Configuration class for TextCleaner to make initialization more structured"""
    language: str = 'english'
    split_methods: Optional[List[str]] = None
    custom_stop_words: Optional[Set[str]] = None
    url_standardization_options: Optional[Dict[str, Any]] = None
    processing_strategy: ProcessingStrategy = ProcessingStrategy.AUTO
    batch_size: int = 1000
    memory_limit_percentage: float = 90.0  # Stop processing if memory usage exceeds this percentage
    max_workers: Optional[int] = None  # Default to number of CPUs - 1
    cache_size: int = 10000  # Size of the LRU cache for text processing
    verbose: bool = False
    datetime_format: str = TextCleanerConstants.DEFAULT_DATETIME_FORMAT 
    datetime_threshold: float = 0.8  
    datetime_patterns: List[str] = field(default_factory=lambda: TextCleanerConstants.DATETIME_PATTERNS)
    
    def __post_init__(self):
        # Set default URL options if none provided
        if self.url_standardization_options is None:
            self.url_standardization_options = TextCleanerConstants.URL_OPTIONS.copy()
            
        # Set default max_workers if None
        if self.max_workers is None:
            self.max_workers = max(1, multiprocessing.cpu_count() - 1)
            
        # Validate language
        if self.language not in TextCleanerConstants.SUPPORTED_LANGUAGES:
            logger.warning(f"Language '{self.language}' not in supported languages. Defaulting to English.")
            self.language = 'english'
            
        # Set logging level based on verbose flag
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)


class NLTKManager:
    """Manages NLTK resources and processing"""
    
    def __init__(self, language='english', custom_stop_words=None):
        if not NLTK_AVAILABLE:
            self.stopwords = set()
            self.lemmatizer = None
            return
        
        self.language = language
        self.custom_stop_words = custom_stop_words or set()
        self.stopwords = self._load_stopwords()  # Load stopwords only once
        self.lemmatizer = WordNetLemmatizer()

    @lru_cache(maxsize=1)
    def _load_stopwords(self):
        """Load stopwords for the given language (cached)."""
        try:
            stopwords_set = set(stopwords.words(self.language))
        except Exception as e:
            logger.warning(f"Error loading stopwords for '{self.language}': {e}. Falling back to English.")
            stopwords_set = set(stopwords.words('english'))  # Fallback to English
        
        if self.custom_stop_words:
            stopwords_set.update(self.custom_stop_words)
        
        return stopwords_set

    def process_text(self, text):
        """Tokenize, remove stopwords, and lemmatize text using NLTK."""
        if not NLTK_AVAILABLE or not isinstance(text, str):
            return text

        try:
            words = word_tokenize(text)
            words = [w for w in words if w not in self.stopwords]
            words = [self.lemmatizer.lemmatize(w) for w in words]
            return ' '.join(words)
        except Exception as e:
            logger.warning(f"Error processing text with NLTK: {e}")
            return text


class MemoryMonitor:
    """Monitors memory usage during processing"""
    
    def __init__(self, limit_percentage=90.0):
        self.limit_percentage = limit_percentage
        
    def check_memory(self):
        """Check if memory usage exceeds the limit"""
        memory = psutil.virtual_memory()
        return memory.percent > self.limit_percentage, memory.percent
    
    def log_memory_usage(self):
        """Log current memory usage"""
        memory = psutil.virtual_memory()
        logger.debug(f"Memory usage: {memory.percent:.1f}% (Used: {memory.used / 1024**3:.1f} GB, "
                   f"Available: {memory.available / 1024**3:.1f} GB)")


class URLProcessor:
    """Handles URL standardization and validation"""
    
    def __init__(self, options=None):
        self.options = options or TextCleanerConstants.URL_OPTIONS.copy()
    
    @staticmethod
    @lru_cache(maxsize=1024)  # Cache URL parsing for efficiency
    def _parse_url(url):
        """Cached URL parsing to reduce redundant calculations."""
        return urlparse(url)

    def standardize_url(self, url):
        """Standardize a single URL"""
        if not url or not isinstance(url, str):
            return url
            
        # Handle relative URLs starting with //
        if url.startswith('//'):
            url = f"{self.options['default_scheme']}:{url}"
            
        # Add default scheme if missing
        if not url.startswith(('http://', 'https://')):
            # Check if it's a valid domain-like string
            if re.match(r'^[a-zA-Z0-9][-a-zA-Z0-9.]*\.[a-zA-Z]{2,}', url):
                url = f"{self.options['default_scheme']}://{url}"
                
        try:
            # Parse the URL
            parsed = self._parse_url(url)
            
            # Skip empty URLs
            if not parsed.netloc and not parsed.path:
                return ''
                
            # Apply case normalization
            if self.options.get('normalize_case', True):
                parsed = parsed._replace(netloc=parsed.netloc.lower())
                
            # Remove www if specified
            if self.options.get('remove_www', True) and parsed.netloc.startswith('www.'):
                parsed = parsed._replace(netloc=parsed.netloc[4:])
                
            # Process query parameters
            if parsed.query:
                remove_query_params = self.options.get('remove_query_params')
                required_query_params = self.options.get('required_query_params')
                
                query_dict = dict(urllib.parse.parse_qsl(parsed.query))
                
                if remove_query_params is True or isinstance(remove_query_params, list):
                    if isinstance(remove_query_params, list):
                        # Remove specific query parameters
                        for param in remove_query_params:
                            query_dict.pop(param, None)
                    elif required_query_params:
                        # Keep only required parameters
                        query_dict = {k: v for k, v in query_dict.items() if k in required_query_params}
                    else:
                        # Remove all query parameters
                        query_dict = {}
                        
                    # Rebuild query string
                    query_string = urllib.parse.urlencode(query_dict)
                    parsed = parsed._replace(query=query_string)
                    
            # Remove fragments if specified
            if self.options.get('remove_fragments', True):
                parsed = parsed._replace(fragment='')
                
            # Handle trailing slash
            if self.options.get('remove_trailing_slash', True) and parsed.path.endswith('/') and len(parsed.path) > 1:
                parsed = parsed._replace(path=parsed.path[:-1])
                
            # Reconstruct the URL
            return urlunparse(parsed)
            
        except Exception as e:
            logger.warning(f"Error standardizing URL '{url}': {e}")
            # If URL parsing fails, return original
            return url
    
    def standardize_urls(self, urls):
        """Standardize a list of URLs"""
        if not urls:
            return []
            
        return [self.standardize_url(url) for url in urls]
    
    def extract_links_from_html(self, html_content, base_url=None):
        """Extract links from HTML content"""
        if not html_content:
            return {"navigation": [], "images": [], "scripts": [], "stylesheets": [], "other": []}
            
        links = {
            "navigation": [],  # a, link tags
            "images": [],      # img tags
            "scripts": [],     # script tags
            "stylesheets": [], # link[rel=stylesheet]
            "other": []        # other link types
        }
        
        # Extract href from a and link tags
        href_pattern = re.compile(r'<a[^>]*href=["\'](.*?)["\']', re.IGNORECASE)
        link_pattern = re.compile(r'<link[^>]*href=["\'](.*?)["\']', re.IGNORECASE)
        
        # Extract src from img, script tags
        img_pattern = re.compile(r'<img[^>]*src=["\'](.*?)["\']', re.IGNORECASE)
        script_pattern = re.compile(r'<script[^>]*src=["\'](.*?)["\']', re.IGNORECASE)
        
        # Extract stylesheet links
        stylesheet_pattern = re.compile(r'<link[^>]*rel=["\'](stylesheet)["\'][^>]*href=["\'](.*?)["\']|<link[^>]*href=["\'](.*?)["\'][^>]*rel=["\'](stylesheet)["\']', re.IGNORECASE)
        
        # Process navigation links
        for match in href_pattern.finditer(html_content):
            url = match.group(1)
            if base_url:
                url = urljoin(base_url, url)
            links["navigation"].append(url)
        
        # Process general link tags
        for match in link_pattern.finditer(html_content):
            url = match.group(1)
            if base_url:
                url = urljoin(base_url, url)
            # Skip stylesheet links as they're handled separately
            if not re.search(r'rel=["\'](stylesheet)["\']', match.group(0), re.IGNORECASE):
                links["other"].append(url)
        
        # Process image links
        for match in img_pattern.finditer(html_content):
            url = match.group(1)
            if base_url:
                url = urljoin(base_url, url)
            links["images"].append(url)
        
        # Process script links
        for match in script_pattern.finditer(html_content):
            url = match.group(1)
            if base_url:
                url = urljoin(base_url, url)
            links["scripts"].append(url)
        
        # Process stylesheet links
        for match in stylesheet_pattern.finditer(html_content):
            url = match.group(2) or match.group(3)
            if base_url:
                url = urljoin(base_url, url)
            links["stylesheets"].append(url)
        
        # Standardize all extracted URLs
        for category in links:
            links[category] = self.standardize_urls(links[category])
            # Remove duplicates while preserving order
            links[category] = list(dict.fromkeys(links[category]))
        
        return links
    
    def check_broken_links(self, urls, additional_checks=False):
        """Check for potentially broken links based on URL structure"""
        issues = []
        
        for url in urls:
            if not url:
                continue
                
            url_issues = []
            
            # Check for scheme
            if not url.startswith(('http://', 'https://')):
                url_issues.append("Missing or invalid scheme")
            
            # Check for malformed URLs
            try:
                parsed = self._parse_url(url)
                if not parsed.netloc:
                    url_issues.append("Missing domain")
            except Exception:
                url_issues.append("Malformed URL")
                
            # Additional validation checks
            if additional_checks:
                # Check for unusual TLDs
                if parsed.netloc and '.' in parsed.netloc:
                    tld = parsed.netloc.split('.')[-1].lower()
                    uncommon_tlds = ['xyz', 'tk', 'ml', 'ga', 'cf', 'gq']
                    if tld in uncommon_tlds:
                        url_issues.append(f"Uncommon TLD (.{tld})")
                
                # Check for extremely long URLs (potential issue)
                if len(url) > 2000:
                    url_issues.append("Extremely long URL")
                    
                # Check for unusual characters in domain
                if parsed.netloc and re.search(r'[^a-zA-Z0-9.-]', parsed.netloc):
                    url_issues.append("Unusual characters in domain")
            
            if url_issues:
                issues.append({
                    "url": url,
                    "issues": url_issues
                })
        
        return issues

class DatetimeProcessor:
    """Handles datetime detection and standardization in both columns and inside text fields."""
    
    def __init__(self, config: CleanerConfig):
        self.output_format = config.datetime_format
        self.threshold = config.datetime_threshold
        self.patterns = config.datetime_patterns  # Use patterns from CleanerConfig

    def standardize_datetime_column(self, series):
        """Standardize a datetime column"""
        parsed_series = pd.to_datetime(series,format=self.output_format, errors='coerce')
        return parsed_series.dt.strftime(self.output_format).fillna('')
    
    def detect_datetime_column(self, series):
        """Detect if a column contains datetime values based on a threshold."""
        if series.empty or not pd.api.types.is_string_dtype(series):
            return False

        # Sample a limited number of rows for efficiency
        sample_size = min(len(series), 1000)
        sample = series.dropna().sample(sample_size, random_state=42)

        # Convert values to datetime, setting errors='coerce' to convert invalid values to NaT
        parsed_sample = pd.to_datetime(sample, format=self.output_format, errors='coerce')

        # Calculate percentage of valid datetime values
        valid_percentage = parsed_sample.notna().sum() / sample_size

        return valid_percentage >= self.threshold

    def standardize_datetime_in_text(self, text):
        """Find and standardize datetime patterns inside text content."""
        if not isinstance(text, str) or not text.strip():
            return text
        
        def replace_match(match):
            date_str = match.group(1)
            try:
                parsed_date = dateutil_parse(date_str)
                return parsed_date.strftime(self.output_format)
            except Exception:
                return date_str  # Return original if parsing fails
        
        for pattern in self.patterns:
            text = re.sub(pattern, replace_match, text)
        
        return text

    def detect_and_standardize_datetimes(self, df, text_columns=None):
        """
        Detect datetime columns and also process datetime values embedded in text content.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            text_columns (list): List of columns to search for datetime inside text.
            
        Returns:
            pd.DataFrame: DataFrame with datetime fields and text-based datetime values standardized.
        """
        new_df = df.copy()

        # Step 1: Detect & Standardize Date Columns
        for col in new_df.columns:
            if new_df[col].dtype == 'object' or pd.api.types.is_string_dtype(new_df[col]):
                if self.detect_datetime_column(new_df[col]):
                    logger.info(f"Column '{col}' detected as datetime and will be standardized")
                    new_df[col] = self.standardize_datetime_column(new_df[col])
                else:
                    logger.debug(f"Column '{col}' not detected as datetime")

        # Step 2: Standardize Datetime Inside Text Columns
        if text_columns:
            for col in text_columns:
                if col in new_df.columns and pd.api.types.is_string_dtype(new_df[col]):
                    logger.info(f"Checking '{col}' for embedded datetime values")
                    new_df[col] = new_df[col].apply(self.standardize_datetime_in_text)
        
        return new_df

class TextCleaner:
    """Enhanced text cleaner for data preprocessing"""
    
    def __init__(self, config=None):
        """Initialize the TextCleaner with optional configuration"""
        self.config = config or CleanerConfig()
        if isinstance(self.config, list):
            raise ValueError("Config must be a CleanerConfig object, not a list")
        self.nltk_manager = NLTKManager(self.config.language, self.config.custom_stop_words)
        self.url_processor = URLProcessor(self.config.url_standardization_options)
        self.datetime_processor = DatetimeProcessor(self.config)
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_percentage)
        
        # Set up LRU cache for text cleaning
        self.clean_text_cached = lru_cache(maxsize=self.config.cache_size)(self._clean_text_impl)
        
        logger.info(f"TextCleaner initialized with {self.config.language} language support")
        logger.info(f"Processing strategy: {self.config.processing_strategy.value}")
        
        # Log availability of optional dependencies
        if NLTK_AVAILABLE:
            logger.info("NLTK is available for advanced text processing")
        else:
            logger.warning("NLTK is not available - some features disabled")
            
        if DATEUTIL_AVAILABLE:
            logger.info("dateutil is available for datetime processing")
        else:
            logger.warning("dateutil is not available - datetime processing disabled")
            
        if WORD_SPLITTER_AVAILABLE:
            logger.info("word_splitter is available for complex word splitting")
        else:
            logger.warning("word_splitter is not available - word splitting disabled")
            
    def _clean_text_impl(self, text):
        """Implement text cleaning logic (non-cached version)"""
        if not isinstance(text, str):
            return text

        # Remove URLs and replace with placeholder
        text = TextCleanerConstants.URL_PATTERN.sub(" URL ", text)
        
        # Remove emails and replace with placeholder
        text = TextCleanerConstants.EMAIL_PATTERN.sub(" EMAIL ", text)
        
        # Convert to lowercase and replace newlines with spaces
        text = text.lower().replace('\n', " ")

        # Apply word splitter if methods are specified
        if self.config.split_methods and WORD_SPLITTER_AVAILABLE:
            try:
                text = general_word_splitter(text, self.config.split_methods)
            except Exception as e:
                logger.warning(f"Error using word splitter: {e}")

        # Ensure space after punctuation
        text = TextCleanerConstants.PUNCTUATION_SPACE_RE.sub(r"\1 ", text)

        # Remove non-ASCII characters
        text = TextCleanerConstants.NON_ASCII_RE.sub(" ", text)

        # Remove extra spaces
        text = TextCleanerConstants.EXTRA_SPACE_RE.sub(" ", text).strip()

        # Remove single characters
        text = TextCleanerConstants.SINGLE_CHAR_RE.sub("", text)

        return text.strip()
        
    def clean_text(self, text):
        """Clean text by applying various text processing techniques (cached version)"""
        return self.clean_text_cached(text)
    
    def process_with_nltk(self, text):
        """Process text with NLTK"""
        return self.nltk_manager.process_text(text)
        
    def _check_memory_usage(self):
        """Check memory usage and log if needed"""
        limit_exceeded, percentage = self.memory_monitor.check_memory()
        if limit_exceeded:
            logger.warning(f"Memory usage critical: {percentage:.1f}% (limit: {self.config.memory_limit_percentage}%)")
        elif self.config.verbose:
            self.memory_monitor.log_memory_usage()
        return limit_exceeded
    
    def _process_single_text(self, text, apply_nltk):
        """Clean and optionally apply NLTK to a single text."""
        cleaned = self.clean_text(text)
        if apply_nltk and NLTK_AVAILABLE:
            cleaned = self.process_with_nltk(cleaned)
        return cleaned

    def process_text_batch(self, texts, apply_nltk=True):
        """Process a batch of texts with chosen strategy."""
        if not texts:
            return []
            
        logger.info(f"Processing {len(texts)} texts")
        
        strategy = self.config.processing_strategy
        if strategy == ProcessingStrategy.AUTO:
            strategy = ProcessingStrategy.MULTIPROCESSING if len(texts) >= 1000 else ProcessingStrategy.SEQUENTIAL
                
        if strategy == ProcessingStrategy.SEQUENTIAL:
            logger.info("Using sequential processing")
            return self._process_text_sequential(texts, apply_nltk)
        elif strategy == ProcessingStrategy.MULTIPROCESSING:
            logger.info(f"Using multiprocessing with {self.config.max_workers} workers")
            return self._process_text_multiprocessing(texts, apply_nltk)
        elif strategy == ProcessingStrategy.VECTORIZE:
            logger.info("Using vectorized processing")
            return self._process_text_vectorized(texts, apply_nltk)
        else:
            raise ValueError(f"Invalid processing strategy: {strategy}")

    def _process_text_sequential(self, texts, apply_nltk):
        """Sequential processing of texts."""
        start_time = time.time()
        cleaned_texts = []
        for i, text in enumerate(texts):
            if i % 1000 == 0 and self._check_memory_usage():
                logger.warning("Processing aborted due to high memory usage")
                break
            cleaned_texts.append(self._process_single_text(text, apply_nltk))
            if i % 1000 == 0 and i > 0:
                elapsed = time.time() - start_time
                progress = min(100, (i + 1) / len(texts) * 100)
                logger.info(f"Processed {progress:.1f}% ({i + 1}/{len(texts)}) in {elapsed:.1f}s")
        logger.info(f"Sequential processing completed in {time.time() - start_time:.1f}s")
        return cleaned_texts

    def _process_text_multiprocessing(self, texts, apply_nltk):
        """Multiprocessing of texts."""
        start_time = time.time()
        cleaned_texts = []
        batch_size = self.config.batch_size
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        logger.info(f"Split {len(texts)} texts into {len(batches)} batches")

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Use partial to pass apply_nltk to the worker function
            func = partial(self._process_batch, apply_nltk=apply_nltk)
            future_to_batch = {executor.submit(func, batch): i for i, batch in enumerate(batches)}

            completed = 0
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    if self._check_memory_usage():
                        logger.warning("Processing aborted due to high memory usage")
                        executor.shutdown(wait=False)
                        break
                    result = future.result()
                    cleaned_texts.extend(result)
                    completed += 1
                    progress = min(100, completed / len(batches) * 100)
                    elapsed = time.time() - start_time
                    logger.info(f"Completed batch {batch_idx+1}/{len(batches)} ({progress:.1f}%) in {elapsed:.1f}s")
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
        logger.info(f"Multiprocessing completed in {time.time() - start_time:.1f}s")
        return cleaned_texts

    def _process_text_vectorized(self, texts, apply_nltk):
        """Vectorized processing of texts using pandas."""
        start_time = time.time()
        series = pd.Series(texts)
        cleaned_series = series.apply(self.clean_text)
        if apply_nltk and NLTK_AVAILABLE:
            cleaned_series = cleaned_series.apply(self.process_with_nltk)
        cleaned_texts = cleaned_series.tolist()
        logger.info(f"Vectorized processing completed in {time.time() - start_time:.1f}s")
        return cleaned_texts
        
    def _process_batch(self, batch, apply_nltk=True):
        """Process a single batch of texts (for multiprocessing)"""
        cleaned_batch = [self._process_single_text(text, apply_nltk) for text in batch]
        return cleaned_batch
        
    def is_url_column(self, col_name):
        """Check if a column name suggests it contains URLs"""
        if not isinstance(col_name, str):
            return False
            
        url_indicators = ['url', 'link', 'href', 'uri', 'web', 'http', 'https', 'www', 'images']
        col_lower = col_name.lower()
        return any(indicator in col_lower for indicator in url_indicators)
    
    def standardize_url(self, url):
        """Standardize a single URL"""
        return self.url_processor.standardize_url(url)
        
    def standardize_urls(self, urls):
        """Standardize a batch of URLs"""
        return self.url_processor.standardize_urls(urls)
        

    def process_dataframe(self, df, text_columns=None, url_columns=None, datetime_columns=None):
        """
        Process a complete DataFrame with text cleaning, URL standardization, and datetime processing
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        text_columns (list): Columns to apply text cleaning (if None, auto-detect)
        url_columns (list): Columns to apply URL standardization (if None, auto-detect)
        datetime_columns (list): Columns to apply datetime standardization (if None, auto-detect)
        
        Returns:
        pd.DataFrame: Processed DataFrame
        """
        if df is None or df.empty:
            return df
            
        logger.info(f"Processing DataFrame with {len(df)} rows and {len(df.columns)} columns")
        result_df = df.copy()
        
        # Auto-detect columns if not specified
        if text_columns is None:
            text_columns = self._detect_text_columns(df)
            logger.info(f"Auto-detected {len(text_columns)} text columns: {text_columns}")
            
        if url_columns is None:
            url_columns = self._detect_url_columns(df)
            logger.info(f"Auto-detected {len(url_columns)} URL columns: {url_columns}")
            
        # Process text columns
        if text_columns:
            for col in text_columns:
                if col in result_df.columns:
                    logger.info(f"Processing text column: {col}")
                    result_df[col] = self.process_text_batch(result_df[col].fillna('').tolist())

        # Process URL columns
        if url_columns:
            for col in url_columns:
                if col in result_df.columns:
                    logger.info(f"Standardizing URL column: {col}")
                    result_df[col] = result_df[col].fillna('').apply(self.standardize_url)
        
        # Process datetime columns using detector
        if datetime_columns is None:
            result_df = self.datetime_processor.detect_and_standardize_datetimes(result_df)
        else:
            for col in datetime_columns:
                if col in result_df.columns:
                    logger.info(f"Standardizing datetime column: {col}")
                    result_df[col] = self.datetime_processor.standardize_datetime_column(result_df[col])
        
        return result_df

    def _detect_text_columns(self, df, min_word_count=3, sample_size=100):
        """Auto-detect text columns based on average word count"""
        text_columns = []

            
        for col in df.select_dtypes(include=['object']).columns:
                # Sample the column
                sample = df[col].dropna().sample(min(sample_size, len(df[col].dropna())))
                if sample.empty:
                    continue
                    
                # Calculate average word count
                avg_word_count = sample.astype(str).str.split().str.len().mean()
                
                # If average word count exceeds threshold, consider it a text column
                if avg_word_count >= min_word_count:
                    text_columns.append(col)
                    
        return text_columns

    def _detect_url_columns(self, df, threshold=0.5, sample_size=100):
        """Auto-detect URL columns based on URL patterns and column names"""
        url_columns = []
        url_pattern = TextCleanerConstants.URL_PATTERN
        
        # First check for URL-suggesting column names
        for col in df.select_dtypes(include=['object']).columns:
            if self.is_url_column(col):
                # Column name suggests URLs, now check content
                sample = df[col].dropna().sample(min(sample_size, len(df[col].dropna())))
                if sample.empty:
                    continue
                    
                # Count URL matches
                url_matches = sample.astype(str).str.match(url_pattern).mean()
                
                if url_matches >= threshold:
                    url_columns.append(col)
                    continue
            
            # For other columns, check for high URL presence
            sample = df[col].dropna().sample(min(sample_size, len(df[col].dropna())))
            if sample.empty:
                continue
                
            # Count URL pattern matches
            url_count = sample.astype(str).str.contains(url_pattern, regex=True).mean()
            
            if url_count >= threshold:
                url_columns.append(col)
                
        return url_columns

    def extract_links_from_html(self, html_content, base_url=None):
        """Extract links from HTML content using URLProcessor"""
        return self.url_processor.extract_links_from_html(html_content, base_url)

    def check_broken_links(self, urls, additional_checks=False):
        """Check for potentially broken links"""
        return self.url_processor.check_broken_links(urls, additional_checks)

    def process_html_links(self, df, html_column, base_url_column=None):
        """
        Extract and process links from an HTML column
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        html_column (str): Column containing HTML content
        base_url_column (str): Column containing base URLs (optional)
        
        Returns:
        pd.DataFrame: DataFrame with extracted link information
        """
        if df is None or df.empty or html_column not in df.columns:
            return None
            
        logger.info(f"Extracting links from HTML column: {html_column}")
        
        # Create result DataFrame
        results = []
        
        # Process each row
        for i, row in df.iterrows():
            html = row[html_column]
            
            # Skip empty/invalid HTML
            if not isinstance(html, str) or not html.strip():
                continue
                
            # Get base URL if available
            base_url = None
            if base_url_column and base_url_column in df.columns:
                base_url = row[base_url_column]
                
            # Extract links
            links = self.extract_links_from_html(html, base_url)
            
            # Create result entry
            entry = {
                'row_id': i,
                'total_links': sum(len(links[k]) for k in links),
            }
            
            # Add counts for each link type
            for link_type, urls in links.items():
                entry[f'{link_type}_count'] = len(urls)
                
                # Join URLs into a single string, removing duplicates while preserving order
                unique_urls = []
                seen = set()
                for url in urls:
                    if url not in seen:
                        unique_urls.append(url)
                        seen.add(url)
                
                entry[f'{link_type}_links'] = ','.join(unique_urls) if unique_urls else ''
                
            results.append(entry)
            
        # Create DataFrame from results
        result_df = pd.DataFrame(results)
        
        logger.info(f"Extracted link information for {len(result_df)} rows")
        return result_df

        