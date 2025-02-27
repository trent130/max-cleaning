import re
import logging
# import os
from functools import lru_cache
import time
#import concurrent
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Use multiprocessing to speed up processing
# from functools import partial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
# Import the splitter functionality
from word_splitter import general_word_splitter

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
        if token.isalpha() and token not in stopwords_set
    ]

    # Apply lemmatization
    lemmatized = [lemmatizer_instance.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)


def _process_text_batch(batch_data):
    """Process a batch of texts - receives a tuple of (texts, stopwords, lemmatizer)"""
    texts, stopwords_set, lemmatizer_instance = batch_data
    return [_process_single_text(text, stopwords_set, lemmatizer_instance) for text in texts]


class TextCleaner:
    """A class for text cleaning operations with optional word splitting."""

    def __init__(self, split_methods=None, custom_stop_words=None):
        """
        Initialize the TextCleaner.

        Args:
            split_methods (list): List of word splitting methods to use
            custom_stop_words (set/list): Additional stopwords to remove
        """
        self.split_methods = split_methods
        self.stopwords = STOPWORDS.copy()

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

        # Without multiprocessing - use this as a fallback if needed
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
    

    def clean_dataframe(self, df, text_columns=None, exclude_columns=None,
                        apply_nltk=True, standardize_datetimes=False, threshold=0.8, output_format="%Y-%m-%d %H:%M:%S"):
        """
        Clean a DataFrame's text columns, and optionally standardize datetime columns.

        Args:
            df (pd.DataFrame): DataFrame to clean
            text_columns (list): List of columns to clean
            exclude_columns (list): List of columns to exclude from cleaning
            apply_nltk (bool): Whether to apply NLTK processing
            standardize_datetimes (bool): Whether to detect and standardize datetime columns
            threshold (float): Fraction of successfully parsed values needed to consider a column as datetime
            output_format (str): The standardized datetime string format

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Create a copy to avoid modifying the original
        df_clean = df.copy()

        if text_columns is None:
            text_columns = [col for col in df_clean.columns if df_clean[col].dtype == 'object']

        if exclude_columns is None:
            exclude_columns = []

        # Remove excluded columns from text_columns
        text_columns = [col for col in text_columns if col not in exclude_columns]

        # Identify URL columns to preserve
        url_columns = [col for col in text_columns if self.is_url_column(col)]
        non_url_columns = [col for col in text_columns if col not in url_columns]

        # Store original URL column values
        url_columns_data = {col: df_clean[col].copy() for col in url_columns}

        # Process non-URL text columns
        logging.info(f"Cleaning text columns: {non_url_columns}")
        for col in non_url_columns:
            try:
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].apply(self.clean_text)
            except Exception as e:
                logging.error(f"Error cleaning column {col}: {e}")

        # Apply NLTK processing if requested
        if apply_nltk and NLTK_AVAILABLE and non_url_columns:
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

        # Restore URL columns
        for col, original_values in url_columns_data.items():
            df_clean[col] = original_values

        # Optional: Standardize datetime columns
        if standardize_datetimes:
            try:
                from datetime_processor import detect_and_standardize_datetimes_custom
                df_clean = detect_and_standardize_datetimes_custom(df_clean, threshold=threshold, output_format=output_format)
                logging.info("Datetime columns standardized.")
            except Exception as e:
                logging.error(f"Error standardizing datetime columns: {e}")

        # Final cleanup: drop rows with NaN, remove duplicates, and reset index
        df_clean.dropna(inplace=True)
        df_clean.drop_duplicates(keep='first', inplace=True)
        df_clean.reset_index(drop=True, inplace=True)


        return df_clean
