import re
import logging
from functools import lru_cache
import time
import urllib.parse
from urllib.parse import urlparse, urljoin, urlunparse
from typing import List, Dict, Union, Optional
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# import pandas as pd
from dateutil.parser import parse

# Use multiprocessing to speed up processing
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
# Import the splitter functionality
from word_splitter import general_word_splitter
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Regex patterns compiled once for efficiency
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
EXTRA_SPACE_RE = re.compile(r"\s+")
SINGLE_CHAR_RE = re.compile(r"\b[A-Za-z]\b")
SENTENCE_END_RE = re.compile(r"([.?!])\s+(?=[A-Z])")
PUNCTUATION_SPACE_RE = re.compile(r"([.,?!])(?! )")

# nltk importation
try:
    NLTK_AVAILABLE = True

    # Download necessary NLTK resources if not present
    @lru_cache(maxsize=1)
    def ensure_nltk_resources():
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)

    ensure_nltk_resources()

    STOPWORDS = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

except ImportError:
    NLTK_AVAILABLE = False
    logging.warning(
        "NLTK not available. Some advanced features will be disabled.")
    STOPWORDS = set()


def _process_single_text(text, stopwords_set, lemmatizer_instance):
    """Process a single text with NLTK"""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Tokenize and filter out stopwords
    tokens = [
        token for token in word_tokenize(text)
        if token.isalpha() # and token not in stopwords_set
    ]

    # Apply lemmatization
    lemmatized = [lemmatizer_instance.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)


def _process_text_batch(batch_data):
    """Process a batch of texts - receives a tuple of (texts, stopwords, lemmatizer)"""
    texts, stopwords_set, lemmatizer_instance = batch_data
    return [_process_single_text(text, stopwords_set, lemmatizer_instance) for text in texts]


def standardize_urls(urls: Union[str, List[str]], 
                     default_scheme: str = 'https',
                     normalize_case: bool = True, 
                     remove_fragments: bool = True,
                     remove_query_params: Union[bool, List[str]] = False,
                     required_query_params: Optional[List[str]] = None,
                     remove_trailing_slash: bool = True,
                     remove_www: bool = True) -> Union[str, List[str]]:
    """
    Standardize URLs by applying consistent formatting rules.
    
    Args:
        urls: A single URL string or a list of URL strings to standardize
        default_scheme: Default scheme to use if none is present (default: 'https')
        normalize_case: Convert domain to lowercase (default: True)
        remove_fragments: Remove URL fragments (default: True)
        remove_query_params: If True, removes all query parameters; if a list, removes only specified parameters
        required_query_params: List of query parameters to keep even when remove_query_params=True
        remove_trailing_slash: Remove trailing slash from path (default: True)
        remove_www: Remove 'www.' from the domain (default: True)
    
    Returns:
        Standardized URL string or list of standardized URL strings depending on input
    """
    single_input = isinstance(urls, str)
    if single_input:
        urls = [urls]
    
    standardized_urls = []
    for url in urls:
        if not url or not isinstance(url, str):
            standardized_urls.append('')
            continue
            
        # Handle relative URLs starting with //
        if url.startswith('//'):
            url = f"{default_scheme}:{url}"
        
        # Add default scheme if missing
        if not url.startswith(('http://', 'https://')):
            # Check if it's a valid domain-like string
            if re.match(r'^[a-zA-Z0-9][-a-zA-Z0-9.]*\.[a-zA-Z]{2,}', url):
                url = f"{default_scheme}://{url}"
        
        try:
            # Parse the URL
            parsed = urlparse(url)
            
            # Skip empty URLs
            if not parsed.netloc and not parsed.path:
                standardized_urls.append('')
                continue
            
            # Apply case normalization
            if normalize_case:
                parsed = parsed._replace(netloc=parsed.netloc.lower())
            
            # Remove www if specified
            if remove_www and parsed.netloc.startswith('www.'):
                parsed = parsed._replace(netloc=parsed.netloc[4:])
            
            # Process query parameters
            if parsed.query and (remove_query_params is True or isinstance(remove_query_params, list)):
                query_dict = dict(urllib.parse.parse_qsl(parsed.query))
                
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
            if remove_fragments:
                parsed = parsed._replace(fragment='')
            
            # Handle trailing slash
            if remove_trailing_slash and parsed.path.endswith('/') and len(parsed.path) > 1:
                parsed = parsed._replace(path=parsed.path[:-1])
            
            # Reconstruct the URL
            standardized_url = urlunparse(parsed)
            standardized_urls.append(standardized_url)
            
        except Exception:
            # If URL parsing fails, return original
            standardized_urls.append(url)
    
    return standardized_urls[0] if single_input else standardized_urls


def extract_links_from_html(html_content: str, base_url: str = None) -> Dict[str, List[str]]:
    """
    Extract all links from HTML content and categorize them by type.
    
    Args:
        html_content: HTML content as string
        base_url: Base URL to resolve relative URLs
    
    Returns:
        Dictionary with categorized links (navigation, images, scripts, stylesheets, etc.)
    """
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
        links[category] = standardize_urls(links[category])
        # Remove duplicates while preserving order
        links[category] = list(dict.fromkeys(links[category]))
    
    return links

def standardize_datetime_in_text(text, output_format="%Y-%m-d %H:%M:%S"):
    """
    Find and standardize patterns within text
    Args:
        text (str): Text that has datetime patterns(probability)
        output_format (str): Format for standardized datetime strings
    
    Returns:
        str: Text with the standardized patterns
    """
    if not isinstance(text, str) or not text:
        return text
    
    # common datetime patterns
    patterns = [
        # ISO format(2025:3:1T03:50:30)
        r'(\d{4}-\d{2}-\d{3}T\d{3}:\d{2}:\d{2})'
        # common date formats(03-01-2025, 01-03-2025, 03.01.2025 )
        r'(\d{1, 2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})'
        # Month name formats(Mar 01, 2025, March 01, 2025
        r'([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})'
        # date with time 03/01/2025 03:57:34
        r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\s+\d{1,2}:\d{1,2}:\d{1,2})'
        # others like Sartuday, March 01, 2025
        r'([A-Za-z]+,\s+[A-Za-z]+\s+\d{1,2},\s+\d{4})'
    ]

    # standardize a single datetime match
    def replace_match(match):
        date_str = match.group(1)
        try:
            parsed_date = parse(date_str)
            return match.group(0).replace(date_str, parsed_date.strftime(output_format))
        except Exception as e:
            return match.group(0)
        
    # process each pattern
    result = text
    for pattern in patterns:
        result = re.sub(pattern, replace_match, result)
    
    return result


def find_broken_links(urls: List[str], additional_checks: bool = False) -> List[Dict]:
    """
    Identify potentially broken links based on URL structure.
    This doesn't actually make HTTP requests but checks for common issues.
    
    Args:
        urls: List of URLs to check
        additional_checks: Whether to perform additional validation checks
    
    Returns:
        List of dictionaries containing problem URLs and issues
    """
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
            parsed = urlparse(url)
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


class TextCleaner:
    """A class for text cleaning operations with optional word splitting and URL standardization."""

    def __init__(self, split_methods=None, custom_stop_words=None, url_standardization_options=None):
        """
        Initialize the TextCleaner.

        Args:
            split_methods (list): List of word splitting methods to use
            custom_stop_words (set/list): Additional stopwords to remove
            url_standardization_options (dict): Options for URL standardization
        """
        self.split_methods = split_methods
        self.stopwords = STOPWORDS.copy()
        
        # Default URL standardization options
        self.url_standardization_options = {
            'default_scheme': 'https',
            'normalize_case': True,
            'remove_fragments': True,
            'remove_query_params': ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content'],
            'required_query_params': None,
            'remove_trailing_slash': True,
            'remove_www': True
        }
        
        # Update with custom options if provided
        if url_standardization_options:
            self.url_standardization_options.update(url_standardization_options)

        if custom_stop_words:
            self.stopwords.update(custom_stop_words)

    def clean_text(self, text):
        """
        Clean text by applying various text processing techniques.

        Args:
            text (str): The text to clean

        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return text

        # Convert to lowercase and replace newlines with spaces
        text = text.lower().replace('\n', " ")

        # Apply word splitter if methods are specified
        if self.split_methods:
            text = general_word_splitter(text, self.split_methods)

        # Ensure space after punctuation
        text = PUNCTUATION_SPACE_RE.sub(r"\1 ", text)

        # Remove non-ASCII characters
        text = NON_ASCII_RE.sub(" ", text)

        # Remove extra spaces
        text = EXTRA_SPACE_RE.sub(" ", text).strip()

        # Remove single characters
        text = SINGLE_CHAR_RE.sub("", text)

        return text.strip()

    def standardize_url(self, url):
        """
        Standardize a single URL using current standardization options.
        
        Args:
            url (str): URL to standardize
            
        Returns:
            str: Standardized URL
        """
        if not url or not isinstance(url, str):
            return url
            
        return standardize_urls(url, **self.url_standardization_options)
    
    def standardize_url_batch(self, urls):
        """
        Standardize a batch of URLs using current standardization options.
        
        Args:
            urls (list): List of URLs to standardize
            
        Returns:
            list: List of standardized URLs
        """
        if not urls:
            return []
            
        return standardize_urls(urls, **self.url_standardization_options)
    
    def extract_links(self, html_content, base_url=None):
        """
        Extract links from HTML content.
        
        Args:
            html_content (str): HTML content
            base_url (str): Base URL for resolving relative URLs
            
        Returns:
            dict: Dictionary of categorized links
        """
        return extract_links_from_html(html_content, base_url)
    
    def check_links(self, urls, additional_checks=False):
        """
        Check for potentially broken links.
        
        Args:
            urls (list): List of URLs to check
            additional_checks (bool): Whether to perform additional validation
            
        Returns:
            list: List of dictionaries with problem URLs and issues
        """
        return find_broken_links(urls, additional_checks)

    def process_with_nltk(self, texts, batch_size=1000):
        """
        Process a list of texts with NLTK for lemmatization and stopword removal.

        Args:
            texts (list): List of texts to process
            batch_size (int): Number of texts to process in each batch

        Returns:
            list: List of processed texts
        """
        if not NLTK_AVAILABLE:
            logging.warning(
                "NLTK not available. Skipping advanced processing.")
            return texts

        # Check if texts is empty
        if not texts:
            return []

        logging.info(f"Processing {len(texts)} texts with NLTK")


        if len(texts) < 1000:  # For small datasets, don't use multiprocessing
            processed_texts = []
            for text in texts:
                processed_texts.append(_process_single_text(
                    text, self.stopwords, lemmatizer))
            return processed_texts

        try:
            # Try with multiprocessing
            # Create batches
            batches = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                # Each batch is a tuple of (texts, stopwords, lemmatizer)
                batches.append((batch, self.stopwords, lemmatizer))

            # Use Process Pool to process batches in parallel
            # Leave one CPU free
            cpu_count = max(1, multiprocessing.cpu_count() - 1)
            logging.info(
                f"Processing {len(batches)} batches with {cpu_count} workers")

            all_processed = []
            with ProcessPoolExecutor(max_workers=cpu_count) as executor:
                results = list(executor.map(_process_text_batch, batches))

                # Flatten results
                for batch_result in results:
                    all_processed.extend(batch_result)

            return all_processed

        except Exception as e:
            # If multiprocessing fails, fall back to sequential processing
            logging.warning(
                f"Multiprocessing failed with error: {e}. Falling back to sequential processing.")

            processed_texts = []
            start_time = time.time()

            # Process in batches to avoid memory issues
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]

                # Process each text in the batch
                for text in batch:
                    processed_texts.append(_process_single_text(
                        text, self.stopwords, lemmatizer))

                # Log progress
                if (i // batch_size) % 5 == 0:
                    elapsed = time.time() - start_time
                    progress = min(100, (i + len(batch)) / len(texts) * 100)
                    logging.info(
                        f"Processed {progress:.1f}% ({i + len(batch)}/{len(texts)}) in {elapsed:.1f}s")

            logging.info(
                f"Sequential processing completed in {time.time() - start_time:.1f}s")
            return processed_texts

    def is_url_column(self, col_name):
        """
        Check if a column name suggests it contains URLs.

        Args:
            col_name (str): Column name to check

        Returns:
            bool: True if column appears to be URL related
        """
        url_indicators = ['url', 'link', 'href', 'uri', 'web', 'http', 'images']
        col_lower = col_name.lower()
        return any(indicator in col_lower for indicator in url_indicators)
    
    def clean_dataframe(self, df, text_columns=None, url_columns=None, exclude_columns=None,
                      apply_nltk=True, standardize_urls=True, standardize_datetimes=True, 
                      threshold=0.8, output_format="%Y-%m-%d %H:%M:%S", preserve_datetime_text=True):
        """
        Clean a DataFrame's text columns, standardize URLs, and standardize datetime columns.

        Args:
            df (pd.DataFrame): DataFrame to clean
            text_columns (list): List of columns to clean as text
            url_columns (list): List of columns to standardize as URLs
            exclude_columns (list): List of columns to exclude from cleaning
            apply_nltk (bool): Whether to apply NLTK processing
            standardize_urls (bool): Whether to standardize URL columns
            standardize_datetimes (bool): Whether to detect and standardize datetime columns
            threshold (float): Fraction of successfully parsed values needed to consider a column as datetime
            output_format (str): The standardized datetime string format
            preserve_datetime_text (bool): whether to preserve sorrounding text when standardizing dates

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Create a copy to avoid modifying the original
        df_clean = df.copy()

        if exclude_columns is None:
            exclude_columns = []

        # Auto-detect text columns if not provided
        if text_columns is None:
            text_columns = [col for col in df_clean.columns if df_clean[col].dtype == 'object']
            # Remove excluded columns
            text_columns = [col for col in text_columns if col not in exclude_columns]

        # Auto-detect URL columns if not provided
        if url_columns is None:
            url_columns = [col for col in text_columns if self.is_url_column(col)]
        else:
            # Make sure URL columns are not in excluded columns
            url_columns = [col for col in url_columns if col not in exclude_columns]

        # Remove URL columns from text_columns to avoid double processing
        non_url_columns = [col for col in text_columns if col not in url_columns]

        # step1: Process URL columns
        if standardize_urls and url_columns:
            logging.info(f"Standardizing URL columns {url_columns}")
            for col in url_columns:
                try:
                    df_clean[col] = df_clean[col].astype(str)
                    
                    # apply the standardizing
                    df_clean[col] = df_clean[col].apply(self.standardize_url)
                except Exception as e:
                    logging.error(f"Error standardizing URLs in column{col}: {e}")

        # step2: standardize datetime
        if standardize_datetimes:
            if preserve_datetime_text:
                # first identify and replace datetime patterns within text
                for col in non_url_columns:
                    try:
                        df_clean[col] = df_clean[col].apply(lambda x: standardize_datetime_in_text(x, output_format))
                    except Exception as e:
                        logging.error(f"Error standardizing datetime patterns in column {col} {e}")
                    
            else:
                # use the existing detect_standardize_datetimes_custom_function
                # threshold if met, the entire column is treated as datetime
                try:
                    from datetime_processor import detect_and_standardize_datetimes_custom
                    df_clean = detect_and_standardize_datetimes_custom(
                        df_clean,
                        threshold=threshold,
                        output_format=output_format
                    )
                    logging.info("Datetime columns standardized.")
                except Exception as e:
                    logging.error(f"Error standardizing datetime columns: {e}")

        logging.info("Finished standardizing the datetimes in the dataset.")


        # step3: Process non-URL text columns
        if non_url_columns:
            logging.info(f"Cleaning text columns: {non_url_columns}")
            for col in non_url_columns:
                try:
                    df_clean[col] = df_clean[col].astype(str)
                    df_clean[col] = df_clean[col].apply(self.clean_text)
                except Exception as e:
                    logging.error(f"Error cleaning column {col}: {e}")

            # Apply NLTK processing if requested
            if apply_nltk and NLTK_AVAILABLE:
                for col in non_url_columns:
                    try:
                        texts = df_clean[col].tolist()
                        processed_texts = self.process_with_nltk(texts)
                        if len(processed_texts) != len(texts):
                            logging.error(f"Length mismatch in column {col}: original {len(texts)} vs processed {len(processed_texts)}")
                            continue
                        df_clean[col] = processed_texts
                    except Exception as e:
                        logging.error(f"Error applying NLTK to column {col}: {e}")

        # Final cleanup: drop rows with NaN, remove duplicates, and reset index
        df_clean.dropna(inplace=True)
        df_clean.drop_duplicates(keep='first', inplace=True)
        df_clean.reset_index(drop=True, inplace=True)

        return df_clean
    
    def extract_and_clean_html_links(self, html_content, base_url=None):
        """
        Extract links from HTML content and standardize them.
        
        Args:
            html_content (str): HTML content
            base_url (str): Base URL for resolving relative URLs
            
        Returns:
            dict: Dictionary of categorized and standardized links
        """
        # Extract links
        links = self.extract_links(html_content, base_url)
        
        # Standardize links in each category
        for category in links:
            links[category] = self.standardize_url_batch(links[category])
            
        return links
    
    def html_link_report(self, df, html_column, base_url_column=None):
        """
        Generate a report of links found in HTML content.
        
        Args:
            df (pd.DataFrame): DataFrame containing HTML content
            html_column (str): Column name containing HTML content
            base_url_column (str): Column name containing base URLs (optional)
            
        Returns:
            pd.DataFrame: DataFrame with link report
        """
        
        # Initialize empty lists for report data
        row_indices = []
        categories = []
        urls = []
        issues = []
        
        # Process each row
        for idx, row in df.iterrows():
            html = row[html_column]
            base_url = row[base_url_column] if base_url_column else None
            
            # Extract links
            link_dict = self.extract_and_clean_html_links(html, base_url)
            
            # Check for issues
            for category, link_list in link_dict.items():
                link_issues = self.check_links(link_list)
                
                # Add each link to the report
                for link in link_list:
                    row_indices.append(idx)
                    categories.append(category)
                    urls.append(link)
                    
                    # Find issues for this link
                    link_issue = next((i for i in link_issues if i['url'] == link), None)
                    if link_issue:
                        issues.append(', '.join(link_issue['issues']))
                    else:
                        issues.append('')
        
        # Create report DataFrame
        report = pd.DataFrame({
            'row_index': row_indices,
            'category': categories,
            'url': urls,
            'issues': issues
        })
        
        return report