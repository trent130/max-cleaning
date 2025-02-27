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

def general_word_splitter(text, methods=None):
    """
    General-purpose function to split concatenated words using multiple methods.
    """
    if not isinstance(text, str):
        return text
        
    if methods is None:
        methods = ['camel', 'snake', 'kebab', 'number', 'team_names']

    result = text
    if 'team_names' in methods:
        # Pattern for "the" attached to capitalized words
        result = re.sub(r'the([A-Z][a-z]+)', r'the \1', result)
        
        # Pattern for adjacent capitalized words without spaces
        result = re.sub(r'([A-Z][a-z]+)([A-Z][a-z]+)', r'\1 \2', result)
    
    for method in methods:
        if method == 'camel':
            result = split_camel_case(result)
        elif method == 'snake':
            result = split_snake_case(result)
        elif method == 'kebab':
            result = split_kebab_case(result)
        elif method == 'number':
            result = split_number_transitions(result)
    
    # Clean up extra spaces and normalize
    return ' '.join(result.split())

def football_text_cleaner(text):
    """
    Specialized text cleaner for football datasets with specific patterns.
    """
    if not isinstance(text, str):
        return text
    
    result = text
    
    # Pattern for "the" attached to capitalized words
    result = re.sub(r'the([A-Z][a-z]+)', r'the \1', result)
    
    # Split adjacent capitalized words
    result = re.sub(r'([A-Z][a-z]+)([A-Z][a-z]+)', r'\1 \2', result)
    
    # Handle camelCase and PascalCase
    result = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', result)
    
    # Insert space between parentheses and text content
    result = re.sub(r'\)([A-Za-z])', r') \1', result)
    result = re.sub(r'([A-Za-z])\(', r'\1 (', result)
    
    # Handle hyphenated dates/ranges - keep hyphen but ensure spaces around it
    result = re.sub(r'(\d+)-(\d+)', r'\1 - \2', result)

    # Remove unnecessary symbols and excessive spaces
    result = re.sub(r'\s+', ' ', result)  # Remove extra spaces
    result = re.sub(r'(\s)[,.!?]', r'\1', result)  # Remove spaces before punctuation

    result = re.sub(r'\b(49 ers)\b', '49ers', result)
    
    # Insert space between numbers and letters
    result = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', result)
    result = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', result)
    
    # Handle special football abbreviations like "49ers"
    result = re.sub(r'(\d+)([a-z]+)(?!\s)', r'\1\2', result)
    
    # Handle special characters like asterisks
    result = re.sub(r'([*&])([A-Za-z])', r'\1 \2', result)
    result = re.sub(r'([A-Za-z])([*&])', r'\1 \2', result)

    # handle year ranges
    result = re.sub(r'\((\d{4})\s*–\s*(\d{4})\)', r'\1-\2', result)  # Clean year ranges
    
    # Clean up extra spaces and normalize
    return ' '.join(result.split())