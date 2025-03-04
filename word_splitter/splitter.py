import re

def split_camel_case(text):
    """
    Split camelCase or PascalCase text into separate words.
    """
    if not isinstance(text, str):
        return text
    # Handle acronyms and normal camel case transitions
    pattern = r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])'
    return re.sub(pattern, ' ', text)

def split_snake_case(text):
    """
    Split snake_case text into separate words.
    """
    if not isinstance(text, str):
        return text
    return text.replace('_', ' ')

def split_kebab_case(text):
    """
    Split kebab-case text into separate words.
    """
    if not isinstance(text, str):
        return text
    return text.replace('-', ' ')

def split_number_transitions(text):
    """
    Split transitions between numbers and letters.
    """
    if not isinstance(text, str):
        return text
    # Insert space between numbers and letters
    return re.sub(r'(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)', ' ', text)

def split_team_names(text):
    """
    Split patterns commonly found in team names.
    """
    if not isinstance(text, str):
        return text
    result = text
    # Pattern for "the" attached to capitalized words
    result = re.sub(r'the([A-Z][a-z]+)', r'the \1', result)
    # Pattern for adjacent capitalized words without spaces
    result = re.sub(r'([A-Z][a-z]+)([A-Z][a-z]+)', r'\1 \2', result)
    return result

def general_word_splitter(text, methods=None):
    """
    General-purpose function to split concatenated words using multiple methods.
    
    Args:
        text (str): The text to be processed
        methods (list): List of methods to apply. Available options:
            'camel', 'snake', 'kebab', 'number', 'team_names'
            
    Returns:
        str: Processed text with words properly split
    """
    if not isinstance(text, str):
        return text
        
    if methods is None:
        methods = ['camel', 'snake', 'kebab', 'number', 'team_names']
    
    result = text
    
    # Apply each method in sequence
    for method in methods:
        if method == 'camel':
            result = split_camel_case(result)
        elif method == 'snake':
            result = split_snake_case(result)
        elif method == 'kebab':
            result = split_kebab_case(result)
        elif method == 'number':
            result = split_number_transitions(result)
        elif method == 'team_names':
            result = split_team_names(result)
    
    # Clean up extra spaces and normalize
    return ' '.join(result.split())

def football_text_cleaner(text):
    """
    Specialized text cleaner for football datasets with specific patterns.
    
    Args:
        text (str): The football-related text to be cleaned
        
    Returns:
        str: Cleaned and normalized text with proper spacing
    """
    if not isinstance(text, str):
        return text
        
    result = text
    
    # Apply general word splitting methods first
    result = general_word_splitter(result)
    
    # Additional football-specific patterns
    
    # Handle parentheses spacing
    result = re.sub(r'\)([A-Za-z])', r') \1', result)
    result = re.sub(r'([A-Za-z])\(', r'\1 (', result)
    
    # Handle hyphenated dates/ranges - keep hyphen but ensure spaces around it
    result = re.sub(r'(\d+)-(\d+)', r'\1 - \2', result)
    
    # Remove unnecessary symbols and excessive spaces
    result = re.sub(r'\s+', ' ', result)  # Remove extra spaces
    result = re.sub(r'(\s)[,.!?]', r'\1', result)  # Remove spaces before punctuation
    
    # Handle special football abbreviations
    result = re.sub(r'\b(49 ers)\b', '49ers', result)
    
    # Preserve special team names with numbers (like 49ers)
    result = re.sub(r'(\d+)([a-z]+)(?!\s)', r'\1\2', result)
    
    # Handle special characters like asterisks
    result = re.sub(r'([*&])([A-Za-z])', r'\1 \2', result)
    result = re.sub(r'([A-Za-z])([*&])', r'\1 \2', result)
    
    # Standardize year ranges
    result = re.sub(r'\((\d{4})\s*–\s*(\d{4})\)', r'(\1-\2)', result)
    
    # Clean up extra spaces and normalize
    return ' '.join(result.split())