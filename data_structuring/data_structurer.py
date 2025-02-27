import pandas as pd
import urllib.parse
import os
import ast
import logging
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataStructurer:
    def __init__(self):
        pass

    @staticmethod
    def load_and_clean_dataset(file_path, text_columns=None, url_column=None, columns_mapping=None):
        """
        Load and clean a dataset from a file and structure it.

        Args:
            file_path (str): Path to the data file (assumed to be CSV).
            text_columns (list, optional): List of column names to apply text cleaning on.
            url_column (str, optional): Name of the column that contains URLs.
            columns_mapping (dict, optional): Mapping of original column names to new names for output.
                For example: {'title': 'Title', 'summary': 'Summary', ...}

        Returns:
            pd.DataFrame: The cleaned (and structured) DataFrame, or None on error.
        """
        try:
            # Load dataset (assumes CSV format; modify if needed)
            dataset = pd.read_csv(file_path)
            
            def clean_text(text):
                """Safely attempts to parse the string as a literal list and join its elements."""
                if isinstance(text, str):
                    # Simple check for list-like strings
                    if text.startswith('[') and text.endswith(']'):
                        try:
                            # Try simplified string splitting for basic lists
                            items = text[1:-1].split(',')
                            return ' '.join(item.strip().strip("'\"") for item in items)
                        except Exception as e:
                            logging.warning(f"Failed simplified parsing: {e}")
                            return text
                    return text
                return text

            # Apply text cleaning if columns are specified
            if text_columns:
                for col in text_columns:
                    if col in dataset.columns:
                        dataset[col] = dataset[col].apply(clean_text)
                        # Replace 'nan' or empty list-like strings with None
                        dataset[col] = dataset[col].apply(lambda x: None if pd.isna(x) or str(x).lower() == 'nan' or str(x).strip() == '[]' else x)
                    else:
                        logging.warning(f"Text column '{col}' not found in {file_path}.")

            # Remove leading zeros in string columns (if applicable)
            for column in dataset.select_dtypes(include=['object']).columns:
                dataset[column] = dataset[column].apply(lambda x: str(x).lstrip('0') if isinstance(x, str) else x)

            # URL validation if a URL column is specified
            if url_column and url_column in dataset.columns:
                def is_valid_url(url):
                    try:
                        result = urllib.parse.urlparse(url)
                        return all([result.scheme, result.netloc])
                    except ValueError:
                        return False
                dataset[f"{url_column}_valid"] = dataset[url_column].apply(is_valid_url)
            elif url_column:
                logging.warning(f"URL column '{url_column}' not found in {file_path}.")

            # Structure the dataset using columns_mapping if provided
            if columns_mapping:
                missing = [k for k in columns_mapping.keys() if k not in dataset.columns]
                if missing:
                    logging.warning(f"The following expected columns are missing in {file_path}: {missing}")
                # Only keep and rename the columns that exist
                valid_mapping = {k: v for k, v in columns_mapping.items() if k in dataset.columns}
                structured_dataset = dataset[list(valid_mapping.keys())].rename(columns=valid_mapping)
            else:
                structured_dataset = dataset

            return structured_dataset

        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            logging.error(f"An error occurred while processing {file_path}: {e}")
            return None

    def process_data_directory(self, data_directory, text_columns=None, url_column=None, columns_mapping=None, output_subdir='structured_data'):
        """
        Process all data files in the given directory:
            - Load each file.
            - Clean and structure the dataset.
            - Export the structured dataset to a designated subdirectory.
        
        Args:
            data_directory (str): Path to the directory containing data files.
            text_columns (list, optional): Columns to clean.
            url_column (str, optional): Column that contains URLs.
            columns_mapping (dict, optional): Mapping of columns for structuring.
            output_subdir (str): Name of the subdirectory (inside data_directory) to save output files.
        
        Returns:
            None
        """
        # Create output subdirectory inside the data directory
        output_directory = os.path.join(data_directory, output_subdir)
        os.makedirs(output_directory, exist_ok=True)

        # Find all files in the data directory
        all_files = glob.glob(os.path.join(data_directory, '*'))
        for file_path in all_files:
            if os.path.isfile(file_path):
                logging.info(f"Processing file: {file_path}")
                # Call the static method correctly using the class name
                structured_dataset = DataStructurer.load_and_clean_dataset(
                    file_path,
                    text_columns=text_columns,
                    url_column=url_column,
                    columns_mapping=columns_mapping
                )
                if structured_dataset is not None:
                    file_name = os.path.basename(file_path)
                    name, ext = os.path.splitext(file_name)
                    output_file_name = f'structured_{name}.csv'
                    output_file_path = os.path.join(output_directory, output_file_name)
                    structured_dataset.to_csv(output_file_path, index=False)
                    logging.info(f"Dataset saved to {output_file_path}")
                else:
                    logging.warning(f"Skipping file due to errors: {file_path}")
            else:
                logging.info(f"Skipping non-file entry: {file_path}")
