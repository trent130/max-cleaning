import re
import logging
import os

# Import the splitter functionality
from word_splitter import general_word_splitter

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Regex patterns compiled once for efficiency
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
EXTRA_SPACE_RE = re.compile(r"\s+")
SINGLE_CHAR_RE = re.compile(r"\b[A-Za-z]\b")
SENTENCE_END_RE = re.compile(r"([.?!])\s+(?=[A-Z])")
PUNCTUATION_SPACE_RE = re.compile(r"([.,?!])(?! )")

# Optional spaCy integration
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
    except OSError:
        logging.warning("Spacy model not found. Downloading...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
    STOPWORDS = set(spacy.lang.en.stop_words.STOP_WORDS)
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Some advanced features will be disabled.")
    STOPWORDS = set()

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
    
    def process_with_spacy(self, texts):
        """
        Process a list of texts with spaCy for lemmatization and stopword removal.
        
        Args:
            texts (list): List of texts to process
            
        Returns:
            list: List of processed texts
        """
        if not SPACY_AVAILABLE:
            logging.warning("spaCy not available. Skipping advanced processing.")
            return texts
        
        logging.info(f"Processing {len(texts)} texts with spaCy")
        
        processed_texts = []
        for doc in nlp.pipe(texts, batch_size=200):
            tokens = [
                token.lemma_ for token in doc
                if token.is_alpha and token.text not in self.stopwords
            ]
            processed_texts.append(" ".join(tokens))
        
        return processed_texts
    
    def is_url_column(self, col_name):
        """
        Check if a column name suggests it contains URLs.
        
        Args:
            col_name (str): Column name to check
            
        Returns:
            bool: True if column appears to be URL related
        """
        url_indicators = ['url', 'link', 'href', 'uri', 'web', 'http']
        col_lower = col_name.lower()
        return any(indicator in col_lower for indicator in url_indicators)
    
    def clean_dataframe(self, df, text_columns=None, exclude_columns=None, 
                        apply_spacy=True):
        """
        Clean a DataFrame's text columns.
        
        Args:
            df (pd.DataFrame): DataFrame to clean
            text_columns (list): List of columns to clean
            exclude_columns (list): List of columns to exclude from cleaning
            apply_spacy (bool): Whether to apply spaCy processing
            
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
                
                # Apply basic text cleaning
                df_clean[col] = df_clean[col].apply(self.clean_text)
                
            except Exception as e:
                logging.error(f"Error cleaning column {col}: {e}")
        
        # Apply spaCy processing if requested
        if apply_spacy and SPACY_AVAILABLE and non_url_columns:
            for col in non_url_columns:
                try:
                    # Collect all texts for batch processing
                    texts = df_clean[col].tolist()
                    
                    # Apply spaCy processing in batch
                    processed_texts = self.process_with_spacy(texts)
                    
                    # Assign results back to the column
                    df_clean[col] = processed_texts
                    
                except Exception as e:
                    logging.error(f"Error applying spaCy to column {col}: {e}")
        
        # Restore URL columns
        for col, original_values in url_columns_data.items():
            df_clean[col] = original_values
        
        return df_clean

