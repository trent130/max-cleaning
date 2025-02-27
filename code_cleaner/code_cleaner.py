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
    
    def __init__(self, max_recursion_depth=10):
        self.max_recursion_depth = max_recursion_depth
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
            # Try to detect file type from extension
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path, encoding=encoding, 
                                  error_bad_lines=False, warn_bad_lines=True,
                                  on_bad_lines='skip', 
                                  encoding_errors=error_handling)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                return pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding=encoding, errors=error_handling) as f:
                    return pd.json_normalize(json.load(f))
            else:
                # Default to CSV for unknown types
                logger.warning(f"Unknown file type for {file_path}, attempting to read as CSV")
                return pd.read_csv(file_path, encoding=encoding, 
                                  encoding_errors=error_handling,
                                  on_bad_lines='skip')
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
            # Read the file line by line
            valid_lines = []
            with open(file_path, 'r', encoding=encoding, errors=error_handling) as f:
                header = f.readline().strip()
                valid_lines.append(header)
                
                for i, line in enumerate(f, 2):
                    try:
                        # Check if the line has the right number of fields
                        if line.count(',') == header.count(','):
                            valid_lines.append(line.strip())
                        else:
                            logger.warning(f"Skipping malformed line {i}: field count mismatch")
                    except Exception as e:
                        logger.warning(f"Error processing line {i}: {str(e)}")
            
            # Create a temporary file with valid lines
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+', encoding=encoding, delete=False) as temp:
                temp.write('\n'.join(valid_lines))
                temp_path = temp.name
            
            # Read the temporary file
            return pd.read_csv(temp_path, encoding=encoding)
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
            
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # If no columns specified, try to detect text columns
        if text_columns is None:
            text_columns = self._detect_text_columns(cleaned_df)
            logger.info(f"Detected text columns: {text_columns}")
        
        for col in text_columns:
            if col in cleaned_df.columns:
                logger.info(f"Cleaning column: {col}")
                cleaned_df[col] = cleaned_df[col].apply(lambda x: self._clean_text_with_code(x) if pd.notna(x) else x)
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
        
        # Sample rows for faster processing
        sample_df = df.sample(min(sample_size, len(df)))
        
        for col in df.select_dtypes(include=['object']).columns:
            # Check if column has strings with code-like patterns
            has_code_patterns = False
            long_text = False
            
            # Calculate mean length of non-null values
            mean_length = sample_df[col].dropna().astype(str).str.len().mean()
            long_text = mean_length > min_length
            
            # Check for code patterns in sampled rows
            for val in sample_df[col].dropna().astype(str):
                if self._has_code_patterns(val):
                    has_code_patterns = True
                    break
            
            if long_text or has_code_patterns:
                text_columns.append(col)
        
        return text_columns
    
    def _has_code_patterns(self, text):
        """
        Check if text contains patterns typically found in code.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if code patterns detected
        """
        # Common patterns found in code
        code_patterns = [
            r'```python', r'```java', r'```javascript', r'```r', r'```sql',  # Markdown code blocks
            r'def\s+\w+\s*\(', r'function\s+\w+\s*\(',  # Function definitions
            r'class\s+\w+\s*[({:]',  # Class definitions
            r'import\s+[\w.]+', r'from\s+[\w.]+\s+import',  # Import statements
            r'for\s+\w+\s+in\s+', r'while\s*\(',  # Loops
            r'if\s*\(.+\)\s*[:{]', r'else\s*[:{]',  # Conditionals
            r'<script', r'<style', r'<html', r'<body'  # HTML/XML tags
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
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
            # Handle Markdown code blocks
            text = self._clean_markdown_code_blocks(text)
            
            # Handle HTML content
            text = self._clean_html_content(text)
            
            # Handle list-like strings and dictionary-like strings
            text = self._clean_data_structures(text)
            
            return text
        
        except Exception as e:
            logger.warning(f"Error cleaning text: {str(e)}")
            # Return original if cleaning fails
            return text
    
    def _clean_markdown_code_blocks(self, text):
        """
        Extract or clean markdown code blocks.
        
        Args:
            text (str): Text containing markdown code blocks
            
        Returns:
            str: Text with cleaned code blocks
        """
        # Pattern for markdown code blocks
        code_block_pattern = r'```(?:\w+)?\n(.*?)\n```'
        
        # Extract content from code blocks or replace with placeholder
        def clean_block(match):
            code_content = match.group(1).strip()
            # You can return just the code, a placeholder, or a cleaned version
            return "[CODE BLOCK: " + code_content[:30] + "...]" if len(code_content) > 30 else "[CODE BLOCK: " + code_content + "]"
        
        return re.sub(code_block_pattern, clean_block, text, flags=re.DOTALL)
    
    def _clean_html_content(self, text):
        """
        Clean HTML content in text.
        
        Args:
            text (str): Text containing HTML
            
        Returns:
            str: Text with HTML cleaned
        """
        # Check if text contains HTML-like content
        if '<' in text and '>' in text and ('</' in text or '/>' in text):
            try:
                soup = BeautifulSoup(text, 'html.parser')
                # Return text content only
                return soup.get_text(separator=' ')
            except:
                # If BeautifulSoup fails, try simpler approach
                return re.sub(r'<[^>]+>', ' ', text)
        return text
    
    def _clean_data_structures(self, text, depth=0):
        """
        Clean text that appears to contain Python data structures like lists or dicts.
        
        Args:
            text (str): Text to clean
            depth (int): Current recursion depth
            
        Returns:
            str: Cleaned text
        """
        if depth > self.max_recursion_depth:
            return text
            
        # Try to clean list-like strings
        if text.strip().startswith('[') and text.strip().endswith(']'):
            try:
                # Simple approach for list-like strings
                inner_text = text.strip()[1:-1]
                items = []
                
                # Simple parser for comma-separated items
                current_item = ""
                bracket_depth = 0
                quote_char = None
                
                for char in inner_text:
                    if char in "\"'" and not quote_char:
                        quote_char = char
                    elif char == quote_char and current_item[-1:] != '\\':
                        quote_char = None
                    elif char in "[{(" and not quote_char:
                        bracket_depth += 1
                    elif char in "}])" and not quote_char:
                        bracket_depth -= 1
                    elif char == ',' and bracket_depth == 0 and not quote_char:
                        items.append(current_item.strip())
                        current_item = ""
                        continue
                        
                    current_item += char
                
                if current_item:
                    items.append(current_item.strip())
                
                # Clean each item recursively
                cleaned_items = [self._clean_data_structures(item, depth+1) for item in items]
                return " ".join(cleaned_items)
                
            except Exception as e:
                logger.debug(f"Failed to parse list-like string: {str(e)}")
                # Return stripped version of original
                return text.strip()
                
        # Try to clean dict-like strings
        elif text.strip().startswith('{') and text.strip().endswith('}'):
            try:
                # For dict-like strings, extract values
                # This is a simplified approach - for complex nested dicts, we might need more
                values_pattern = r':\s*([^,}]+)'
                values = re.findall(values_pattern, text)
                cleaned_values = [value.strip(' "\'') for value in values]
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
        
        # Load the dataset
        df = self.load_dataset(file_path)
        if df is None:
            logger.error(f"Failed to load dataset: {file_path}")
            return None
            
        # Clean code blocks
        cleaned_df = self.clean_code_blocks(df, text_columns)
        
        # Handle missing values
        cleaned_df = self.handle_missing_values(cleaned_df)
        
        # Save cleaned dataset if output path is provided
        if output_path:
            try:
                # Determine file format from output path
                if output_path.endswith('.csv'):
                    cleaned_df.to_csv(output_path, index=False)
                elif output_path.endswith('.xlsx'):
                    cleaned_df.to_excel(output_path, index=False)
                elif output_path.endswith('.json'):
                    cleaned_df.to_json(output_path, orient='records')
                else:
                    # Default to CSV
                    cleaned_df.to_csv(output_path, index=False)
                    
                logger.info(f"Cleaned dataset saved to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save cleaned dataset: {str(e)}")
        
        return cleaned_df
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to process
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        if df is None:
            return None
            
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Replace empty strings with NaN
        result_df = result_df.replace('', pd.NA)
        
        # Replace 'NULL', 'null', 'NaN', etc. with NaN
        null_values = ['NULL', 'null', 'None', 'none', 'NaN', 'nan', 'NA', 'na']
        result_df = result_df.replace(null_values, pd.NA)
        
        # Count missing values
        missing_counts = result_df.isna().sum()
        logger.info(f"Missing value counts:\n{missing_counts}")
        
        return result_df


# if __name__ == "__main__":
#     cleaner = CodeEmbeddedDataCleaner()
#     cleaned_df = cleaner.clean_tutorial_dataset(
#         "tutorial_dataset.csv",
#         text_columns=["description", "example_code"],
#         output_path="cleaned_tutorial_dataset.csv"
#     )
    
#     if cleaned_df is not None:
#         print("First few rows of cleaned data:")
#         print(cleaned_df.head())