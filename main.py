import os
import logging
import pandas as pd
from data_processor import DataProcessor
from data_structuring import DataStructurer
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Get the current directory where the main runner is located
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
cleaned_dir = os.path.join(data_dir, "cleaned_data")
structured_dir = os.path.join(data_dir, "structured_data")

# Ensure the directories exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(cleaned_dir, exist_ok=True)
os.makedirs(structured_dir, exist_ok=True)

# Step 1: Clean the data using DataProcessor
logging.info("Starting data cleaning process...")
main_processor = DataProcessor(
    split_methods=['camel', 'snake', 'kebab', 'number', 'team_names'],
    custom_stop_words=[''],
)

# Process all files for cleaning
main_processor.process_data_directory(
    data_directory=data_dir,
    output_dir=cleaned_dir,
    exclude_columns=['id', 'timestamp', 'url'],  # Common columns to exclude from cleaning
    apply_nltk=True,
    dropna_threshold=0.3,
    export_formats=['csv'],
   # standardize_urls=True,
)

# Step 2: Structure the cleaned data
logging.info("Starting data structuring process...")
structurer = DataStructurer()

# Get all cleaned files
cleaned_files = glob.glob(os.path.join(cleaned_dir, "*.csv"))

for file_path in cleaned_files:
    try:
        # Read the file to analyze its columns
        df = pd.read_csv(file_path)
        file_name = os.path.basename(file_path)
        
        logging.info(f"Analyzing columns for {file_name}")
        
        # Auto-detect potential text columns (columns with object/string dtype)
        text_columns = list(df.select_dtypes(include=['object']).columns)
        
        # Auto-detect URL columns
        url_columns = [col for col in text_columns if any(url_term in col.lower() 
                                                         for url_term in ['url', 'link', 'href', 'uri'])]
        
        # Generate a simple mapping (original column name to title case)
        columns_mapping = {col: col.replace('_', ' ').title() for col in df.columns}
        
        logging.info(f"Detected text columns: {text_columns}")
        logging.info(f"Detected URL columns: {url_columns}") # Updated logging
        
        # Structure this specific file using the static method
        structured_dataset = DataStructurer.load_and_clean_dataset(
            file_path,
            text_columns=text_columns,
            url_columns=url_columns,
            columns_mapping=columns_mapping
        )
        
        if structured_dataset is not None:
            # Save structured dataset
            name, ext = os.path.splitext(os.path.basename(file_path))
            output_file_name = f'structured_{name}.csv'
            output_file_path = os.path.join(structured_dir, output_file_name)
            structured_dataset.to_csv(output_file_path, index=False)
            logging.info(f"Structured dataset saved to {output_file_path}")
        else:
            logging.warning(f"Failed to structure {file_name}")
            
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")

logging.info("All processing completed.")