import pandas as pd
from dateutil.parser import parse
import sys
sys.setrecursionlimit(3000) 

def safe_parse(x):
    """Try to parse a datetime string. Return pd.NaT if it fails."""
    try:
        return parse(x)
    except Exception:
        return pd.NaT

def detect_and_standardize_datetimes_custom(df, threshold=0.8, output_format="%Y-%m-%d %H:%M:%S"):
    """
    Detect columns that share datetime-like values using a custom parsing function,
    then standardize them.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): The fraction of successfully converted values needed to label the column as datetime.
        output_format (str): The standardized datetime string format.
        
    Returns:
        pd.DataFrame: A DataFrame where datetime-like columns have been standardized
                      to the given output format.
    """
    new_df = df.copy()
    
    # Loop through all the columns in the DataFrame.
    for col in new_df.columns:
        if new_df[col].dtype == object or pd.api.types.is_string_dtype(new_df[col]):
            # Apply the custom parser to each value in the column.
            parsed_series = new_df[col].apply(safe_parse)
            
            valid_count = parsed_series.notna().sum()
            total_count = len(parsed_series)
            
            print(f"\nProcessing column: {col}")
            print("Parsed values (using dateutil):")
            print(parsed_series)
            print(f"Valid conversions: {valid_count} / {total_count}")
            
            if total_count > 0 and (valid_count / total_count) >= threshold:
                # Replace the column with parsed datetime values.
                new_df[col] = parsed_series
                # Convert datetime objects to a standardized string format.
                new_df[col] = new_df[col].dt.strftime(output_format)
                print(f"Column '{col}' detected as datetime and standardized.")
            else:
                print(f"Column '{col}' NOT detected as datetime (conversion rate below threshold).")
    
    return new_df

