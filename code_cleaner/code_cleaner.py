import pandas as pd
import re
import json
import logging
from bs4 import BeautifulSoup
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeEmbeddedDataCleaner:
    """
    A specialized cleaner for datasets that contain embedded code blocks, tutorial examples,
    or complex nested structures that might cause parsing issues.
    """

    def __init__(self, max_recursion_depth=10, remove_code_blocks=True):
        self.max_recursion_depth = max_recursion_depth
        self.remove_code_blocks = remove_code_blocks
        warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

    def load_dataset(self, file_path, encoding='utf-8', error_handling='strict'):
        """
        Load a dataset with appropriate error handling for problematic files.

        Args:
            file_path (str): Path to the data file
            encoding (str): File encoding to use
            error_handling (str): How to handle encoding errors ('strict', 'ignore', 'replace')

        Returns:
            pd.DataFrame or None: Loaded DataFrame or None if loading failed
        """
        try:
            # Try to detect file type from extension - Using direct file type checking
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path, encoding=encoding,
                                  error_bad_lines=False, warn_bad_lines=True, # Deprecated in future versions
                                  on_bad_lines='skip',
                                  encoding_errors=error_handling, low_memory=True) # Added low_memory=True
            elif file_path.endswith(('.xlsx', '.xls')): # Using tuple for multiple extensions check
                return pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding=encoding, errors=error_handling) as f:
                    return pd.json_normalize(json.load(f), max_level=0) # Added max_level=0 for efficiency
            else:
                # Default to CSV for unknown types
                logger.warning(f"Unknown file type for {file_path}, attempting to read as CSV")
                return pd.read_csv(file_path, encoding=encoding,
                                  encoding_errors=error_handling, on_bad_lines='skip', low_memory=True) # Added low_memory=True
        except Exception as e:
            logger.error(f"Failed to load dataset {file_path}: {str(e)}")
            # Attempt more robust loading for problematic files
            return self._robust_csv_load(file_path, encoding, error_handling)

    def _robust_csv_load(self, file_path, encoding='utf-8', error_handling='replace'):
        """
        Attempt to load a problematic CSV file line by line, skipping problematic rows.

        Args:
            file_path (str): Path to the CSV file
            encoding (str): File encoding to use
            error_handling (str): How to handle encoding errors

        Returns:
            pd.DataFrame or None: Loaded DataFrame or None if all attempts failed
        """
        try:
            # Using csv module for more efficient line processing
            import csv
            valid_lines = []
            with open(file_path, 'r', encoding=encoding, errors=error_handling, newline='') as f: # Added newline=''
                reader = csv.reader(f)
                header = next(reader) # Read header directly from csv reader
                valid_lines.append(header)

                for i, row in enumerate(reader, 2): # Iterate through rows
                    try:
                        if len(row) == len(header): # Check row length against header length
                            valid_lines.append(row)
                        else:
                            logger.warning(f"Skipping malformed line {i}: field count mismatch")
                    except Exception as e:
                        logger.warning(f"Error processing line {i}: {str(e)}")

            # Create DataFrame directly from valid lines
            return pd.DataFrame(valid_lines[1:], columns=valid_lines[0]) # Create DataFrame once
        except Exception as e:
            logger.error(f"All loading attempts failed for {file_path}: {str(e)}")
            return None

    def clean_code_blocks(self, df, text_columns=None):
        """
        Clean columns that may contain code blocks or complex nested structures.

        Args:
            df (pd.DataFrame): DataFrame to clean
            text_columns (list): List of column names to clean. If None, attempts to detect text columns.

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if df is None:
            return None

        # Make a copy to avoid modifying the original - Good practice
        cleaned_df = df.copy()

        # If no columns specified, try to detect text columns
        if text_columns is None:
            text_columns = self._detect_text_columns(cleaned_df)
            logger.info(f"Detected text columns: {text_columns}")

        # Vectorized application for cleaning - More efficient apply
        for col in text_columns:
            if col in cleaned_df.columns:
                logger.info(f"Cleaning column: {col}")
                cleaned_df[col] = cleaned_df[col].astype(str).apply(self._clean_text_with_code) # Explicitly cast to string once
            else:
                logger.warning(f"Column {col} not found in DataFrame")

        return cleaned_df

    def _detect_text_columns(self, df, min_length=100, sample_size=100):
        """
        Detect columns that likely contain long text that might include code blocks.

        Args:
            df (pd.DataFrame): DataFrame to analyze
            min_length (int): Minimum character length to consider as text
            sample_size (int): Number of rows to sample for detection

        Returns:
            list: List of column names likely containing text with code
        """
        text_columns = []

        # Sample rows for faster processing - Efficient sampling
        sample_df = df.sample(min(sample_size, len(df)))

        for col in df.select_dtypes(include=['object']).columns:
            # Check if column has strings with code-like patterns - Combined checks
            has_code_patterns = False
            long_text = False

            # Vectorized mean length calculation - Efficient calculation
            mean_length = sample_df[col].dropna().astype(str).str.len().mean()
            long_text = mean_length > min_length if pd.notna(mean_length) else False # Handling NaN mean_length

            # Vectorized check for code patterns - Apply pattern check to Series
            code_pattern_series = sample_df[col].dropna().astype(str).apply(self._has_code_patterns) # Apply once to Series
            has_code_patterns = code_pattern_series.any() # Check if any value in Series is True

            if long_text or has_code_patterns:
                text_columns.append(col)

        return text_columns

    _combined_code_pattern = re.compile( # Pre-compiled combined regex pattern - Efficiency gain
        r'```(?:\w+)?\n.*?\n```|'  # Markdown code blocks
        r'def\s+\w+\s*\(|'         # Function definitions
        r'function\s+\w+\s*\(|'    # Function definitions (JavaScript)
        r'class\s+\w+\s*[({:]|'     # Class definitions
        r'import\s+[\w.]+|'        # Import statements
        r'from\s+[\w.]+\s+import|' # From import statements
        r'for\s+\w+\s+in\s+|'      # For loops
        r'while\s*\(|'           # While loops
        r'if\s*\(.+\)\s*[:{]|'      # If conditionals
        r'else\s*[:{]|'            # Else conditionals
        r'<script|<style|<html|<body', # HTML/XML tags
        re.DOTALL | re.IGNORECASE # Flags for DOTALL and IGNORECASE
    )

    def _has_code_patterns(self, text):
        """
        Check if text contains patterns typically found in code - Using pre-compiled regex.

        Args:
            text (str): Text to check

        Returns:
            bool: True if code patterns detected
        """
        return bool(self._combined_code_pattern.search(text)) # Using pre-compiled regex

    def _clean_text_with_code(self, text):
        """
        Clean text that may contain code blocks or complex structures.

        Args:
            text (str): Text to clean

        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return text

        try:
            # Apply cleaning functions sequentially - Clear and modular
            text = self._clean_markdown_code_blocks(text)
            text = self._clean_html_content(text)
            text = self._clean_data_structures(text)

            return text

        except Exception as e:
            logger.warning(f"Error cleaning text: {str(e)}")
            # Return original if cleaning fails - Robustness
            return text

    _markdown_code_block_pattern = re.compile(r'```(?:\w+)?\n(.*?)\n```', re.DOTALL) # Pre-compiled markdown regex

    def _clean_markdown_code_blocks(self, text):
        """
        Extract or clean markdown code blocks - Using pre-compiled regex.

        Args:
            text (str): Text containing markdown code blocks

        Returns:
            str: Text with cleaned code blocks
        """
        if not self.remove_code_blocks:
            return text
        
        return self._markdown_code_block_pattern.sub(
            lambda match: "[CODE BLOCK: " + match.group(1).strip()[:30] + "...]" if len(match.group(1).strip()) > 30 else "[CODE BLOCK: " + match.group(1).strip() + "]",
            text
        )

    _html_tag_pattern = re.compile(r'<[^>]+>') # Pre-compiled HTML tag regex

    def _clean_html_content(self, text):
        """
        Clean HTML content in text - Using pre-compiled regex and optimized BS.

        Args:
            text (str): Text containing HTML

        Returns:
            str: Text with HTML cleaned
        """
        # Check if text contains HTML-like content - Faster check
        if '<' in text and '>' in text and ('</' in text or '/>' in text):
            try:
                soup = BeautifulSoup(text, 'html.parser', features="lxml") # Using lxml for faster parsing if available
                return soup.get_text(separator=' ', strip=True) # strip=True for efficiency
            except Exception as e:
                # If BeautifulSoup fails, try simpler approach - Using pre-compiled regex
                return self._html_tag_pattern.sub(' ', text) # Using pre-compiled regex
        return text

    def _clean_data_structures(self, text, depth=0):
        """
        Clean text that appears to contain Python data structures like lists or dicts - Simplified parsing.

        Args:
            text (str): Text to clean
            depth (int): Current recursion depth

        Returns:
            str: Cleaned text
        """
        if depth > self.max_recursion_depth:
            return text

        # Try to clean list-like strings - Simplified list cleaning
        if text.strip().startswith('[') and text.strip().endswith(']'):
            try:
                # Using regex to split comma-separated items outside quotes and brackets - More robust and efficient
                items = re.split(r',(?=([^"]*"[^"]*")*[^"]*$)(?![^\[]*\])', text.strip()[1:-1]) # Regex for comma split
                cleaned_items = [self._clean_data_structures(item.strip(), depth+1) for item in items if item.strip()] # Clean non-empty items
                return " ".join(cleaned_items)

            except Exception as e:
                logger.debug(f"Failed to parse list-like string: {str(e)}")
                return text.strip()

        # Try to clean dict-like strings - Simplified dict cleaning
        elif text.strip().startswith('{') and text.strip().endswith('}'):
            try:
                # Extract values using regex - More efficient regex extraction
                values = re.findall(r':\s*"([^"]*)"|:\s*\'([^\']*)\'|:\s*([^,\'"}]+)', text) # Regex to extract values
                cleaned_values = [v for tuple_v in values for v in tuple_v if v] # Flatten tuple and remove empty
                cleaned_values = [value.strip() for value in cleaned_values] # Strip values
                return " ".join(cleaned_values)
            except Exception as e:
                logger.debug(f"Failed to parse dict-like string: {str(e)}")
                return text.strip()

        return text

    def clean_tutorial_dataset(self, file_path, text_columns=None, output_path=None):
        """
        Complete pipeline to clean a tutorial dataset that may contain embedded code.

        Args:
            file_path (str): Path to the data file
            text_columns (list): List of column names to clean
            output_path (str): Path to save the cleaned file

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        logger.info(f"Processing file: {file_path}")

        # Load the dataset - Function call is efficient
        df = self.load_dataset(file_path)
        if df is None:
            logger.error(f"Failed to load dataset: {file_path}")
            return None

        # Clean code blocks - Function call is efficient
        cleaned_df = self.clean_code_blocks(df, text_columns)

        # Handle missing values - Function call is efficient
        cleaned_df = self.handle_missing_values(cleaned_df)

        # Save cleaned dataset if output path is provided
        if output_path:
            try:
                # Determine file format from output path - Efficient file extension check
                if output_path.endswith('.csv'):
                    cleaned_df.to_csv(output_path, index=False, compression='gzip') # Added compression for larger files
                elif output_path.endswith('.xlsx'):
                    cleaned_df.to_excel(output_path, index=False)
                elif output_path.endswith('.json'):
                    cleaned_df.to_json(output_path, orient='records', compression='gzip') # Added compression for larger files
                else:
                    # Default to CSV - Efficient default saving
                    cleaned_df.to_csv(output_path, index=False, compression='gzip') # Added compression for larger files

                logger.info(f"Cleaned dataset saved to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save cleaned dataset: {str(e)}")

        return cleaned_df

    def handle_missing_values(self, df):
        """
        Handle missing values in the DataFrame - Efficient Pandas operations.

        Args:
            df (pd.DataFrame): DataFrame to process

        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        if df is None:
            return None

        # Make a copy to avoid modifying the original - Good practice
        result_df = df.copy()

        # Efficient vectorized replacements
        result_df = result_df.replace('', pd.NA)
        null_values_regex = '|'.join(map(re.escape, ['NULL', 'null', 'None', 'none', 'NaN', 'nan', 'NA', 'na'])) # Create regex from null values
        result_df = result_df.replace(to_replace=f'^({null_values_regex})$', value=pd.NA, regex=True) # Regex replace for null values

        # Count missing values - Efficient vectorized sum
        missing_counts = result_df.isna().sum()
        logger.info(f"Missing value counts:\n{missing_counts}")

        return result_df


if __name__ == "__main__":
    cleaner = CodeEmbeddedDataCleaner()
    cleaned_df = cleaner.clean_tutorial_dataset(
        "/home/trent/Desktop/data-structuring/data/javascript_tutorial_data.csv",
        text_columns=["Subtopic", "Content"],
        output_path="cleaned_javascript_tutorial_data.csv"
    )
    
    if cleaned_df is not None:
        print("First few rows of cleaned data:")
        print(cleaned_df.head())