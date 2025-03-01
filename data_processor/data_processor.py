import os
from datetime import datetime
import pandas as pd
import logging
from word_cleaner import TextCleaner
import glob


class DataProcessor:
    """A class for loading, cleaning, and exporting data."""
    output_directory = "/home/trent/Desktop/data-structuring/data"
    
    def __init__(self, split_methods=None, custom_stop_words=None, ):
        """
        Initialize the DataProcessor(with the text cleaner class being initialized to be used inside the data processor class).
        
        Args:
            split_methods (list): List of word splitting methods to use
            custom_stop_words (set/list): Additional stopwords to remove
        """
        self.cleaner = TextCleaner(
        split_methods,
        custom_stop_words,     
        url_standardization_options={
            'remove_query_params': ['utm_source', 'ref'],
            'remove_fragments': True
        })
    
    def load_data(self, file_path):
        """
        Load data from a file based on its extension.
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.csv':
                return pd.read_csv(file_path)
            elif file_extension == '.json':
                return pd.read_json(file_path)
            elif file_extension in [".xlsx", ".xls"]:
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {e}")
            return None
    
    def export_data(self, df, filename="cleaned_data", formats=None):
        """
        Export DataFrame to multiple formats.
        
        Args:
            df (pd.DataFrame): DataFrame to export
            filename (str): Base filename for export
            formats (list): List of formats to export to
        """
        if formats is None:
            formats = ['csv', 'json', 'xlsx']
        
        # Create a timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{filename}_{timestamp}"
        
        try:
            if 'csv' in formats:
                df.to_csv(f"{output_filename}.csv", index=False)
                logging.info(f"Data exported as {output_filename}.csv")
            
            if 'json' in formats:
                df.to_json(f"{output_filename}.json", orient="records", indent=4)
                logging.info(f"Data exported as {output_filename}.json")
            
            if 'xlsx' in formats:
                df.to_excel(f"{output_filename}.xlsx", index=False)
                logging.info(f"Data exported as {output_filename}.xlsx")
        
        except Exception as e:
            logging.error(f"Error exporting data: {e}")
    

    def process_file(self, file_path, text_columns=None, exclude_columns=None, 
                    apply_nltk=True, dropna_threshold=0.5, export_formats=None, output_dir=output_directory):
        """
        Process a single data file and save the cleaned version.

        Args:
            file_path (str): Path to the data file.
            text_columns (list): List of columns to clean.
            exclude_columns (list): List of columns to exclude.
            apply_nltk (bool): Whether to apply NLP cleaning.
            dropna_threshold (float): NaN removal threshold.
            export_formats (list): List of formats to export to.
            output_dir (str, optional): Custom directory for saving cleaned files.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        # Load the data
        df = self.load_data(file_path)
        if df is None:
            return None

        # Apply dropna threshold
        if 0 <= dropna_threshold <= 1:
            min_non_na = int(len(df.columns) * (1 - dropna_threshold))
            df = df.dropna(thresh=min_non_na)
            logging.info(f"Dropped rows with more than {dropna_threshold*100}% missing values")
        else:
            logging.warning("dropna_threshold should be between 0 and 1. Skipping dropna operation.")

        # Clean the DataFrame
        df_cleaned = self.cleaner.clean_dataframe(
            df, 
            text_columns=text_columns,
            exclude_columns=exclude_columns,
            apply_nltk=apply_nltk
        )

        if df_cleaned is not None:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.splitext(os.path.basename(file_path))[0]
                cleaned_file_path = os.path.join(output_dir, f"cleaned_{filename}.csv")
                df_cleaned.to_csv(cleaned_file_path, index=False)
                logging.info(f"Dataset saved to {cleaned_file_path}")
            else:
                logging.info("No such output directory skipping export.")

        return df_cleaned

    def process_data_directory(self, data_directory, text_columns=None, apply_nltk=True,
                               dropna_threshold=0.5, export_formats=None, exclude_columns=None, output_dir=None):
        """
        Process all data files in a given directory.

        Args:
            data_directory (str): Path to the directory containing data files.
            output_dir (str): Directory for saving cleaned files.
            text_columns (list): List of columns to clean.
            exclude_columns (list): List of columns to exclude.
            apply_nltk (bool): Whether to apply NLP cleaning.
            dropna_threshold (float): NaN removal threshold.
            export_formats (list): List of formats to export to.
        """
        # Ensure the directory exists
        if not os.path.exists(data_directory):
            logging.warning(f"Directory does not exist: {data_directory}")
            return

        # Find all files in the data directory
        all_files = glob.glob(os.path.join(data_directory, '*'))

        for file_path in all_files:
            if os.path.isfile(file_path):  # Ensure it's a file
                logging.info(f"Processing file: {file_path}")
                self.process_file(
                    file_path,
                    text_columns=text_columns,
                    exclude_columns=exclude_columns,
                    apply_nltk=apply_nltk,
                    dropna_threshold=dropna_threshold,
                    export_formats=export_formats,
                    output_dir=output_dir
                    )
            else:
                logging.info(f"Skipping {file_path} as it is not a file.")
