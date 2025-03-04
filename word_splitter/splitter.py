import re

# Precompile regex patterns for better performance
CAMEL_CASE_PATTERN = re.compile(
    r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')
NUMBER_TRANSITION_PATTERN = re.compile(
    r'(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)')
TEAM_NAME_THE_PATTERN = re.compile(r'the([A-Z][a-z]+)')
ADJACENT_CAPS_PATTERN = re.compile(r'([A-Z][a-z]+)([A-Z][a-z]+)')
PARENTHESES_SPACING_START = re.compile(r'([A-Za-z])\(')
PARENTHESES_SPACING_END = re.compile(r'\)([A-Za-z])')
HYPHENATED_DATES = re.compile(r'(\d+)-(\d+)')
EXCESS_SPACES = re.compile(r'\s+')
SPACE_BEFORE_PUNCT = re.compile(r'(\s)[,.!?]')
SPECIAL_ABBREVIATIONS = re.compile(r'\b(49 ers)\b')
PRESERVE_NUMERIC_TEAMS = re.compile(r'(\d+ers|\d+ERS)')
SPECIAL_CHARS = re.compile(r'([*&])([A-Za-z])|([A-Za-z])([*&])')
YEAR_RANGES = re.compile(r'\((\d{4})\s*–\s*(\d{4})\)')

# New patterns for cleaning problematic text
FLOATING_PUNCTUATION = re.compile(r'\s*([.,;:?!])\s*')
BRACKETS_WITH_NUMBERS = re.compile(r'\[\s*(\d+)\s*\]\s*:\s*(\d+)')
BRACKETS_SIMPLE = re.compile(r'\[\s*(\d+)\s*\]')
QUOTES_SPACING = re.compile(r'``\s*(.*?)\s*\'\'')
MULTI_DOTS = re.compile(r'\.{2,}')
LONELY_PARENS = re.compile(r'\(\s*\)')
MULTIPLE_DASHES = re.compile(r'-{2,}')
STANDALONE_EQUALS = re.compile(r'\s+=+\s+')
STANDALONE_SYMBOLS = re.compile(r'\s+([.,;:?!])\s+')


def split_camel_case(text):
    """
    Split camelCase or PascalCase text into separate words.
    """
    if not isinstance(text, str):
        return text
    return CAMEL_CASE_PATTERN.sub(' ', text)


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
    Split transitions between numbers and letters, preserving special cases.
    """
    if not isinstance(text, str):
        return text

    # First protect special numeric patterns we want to preserve (like team names)
    protected_text = text

    # Protect common team names and numeric identifiers
    team_patterns = ['49ers', '76ers', '3M', '7Eleven', '5G', '4K', '3D']

    # Create a dictionary to store replacements
    replacements = {}

    # Find and temporarily replace these patterns
    for i, pattern in enumerate(team_patterns):
        placeholder = f"__PROTECTED_{i}__"
        if pattern in protected_text:
            replacements[placeholder] = pattern
            protected_text = protected_text.replace(pattern, placeholder)

    # Apply normal number splitting
    protected_text = NUMBER_TRANSITION_PATTERN.sub(' ', protected_text)

    # Restore protected patterns
    for placeholder, original in replacements.items():
        protected_text = protected_text.replace(placeholder, original)

    return protected_text


def split_team_names(text):
    """
    Split patterns commonly found in team names.
    """
    if not isinstance(text, str):
        return text
    result = text
    # Pattern for "the" attached to capitalized words
    result = TEAM_NAME_THE_PATTERN.sub(r'the \1', result)
    # Pattern for adjacent capitalized words without spaces
    result = ADJACENT_CAPS_PATTERN.sub(r'\1 \2', result)
    return result


def clean_academic_references(text):
    """
    Clean academic reference patterns like [12]: 36
    """
    if not isinstance(text, str):
        return text

    # Clean complex bracket references like [12]: 36
    text = BRACKETS_WITH_NUMBERS.sub(r' (Reference \1, p.\2) ', text)

    # Clean simple bracket references like [2]
    text = BRACKETS_SIMPLE.sub(r' (Reference \1) ', text)

    return text


def clean_punctuation(text):
    """
    Clean and normalize punctuation.
    """
    if not isinstance(text, str):
        return text

    # Fix quote spacing
    text = QUOTES_SPACING.sub(r'"\1"', text)

    # Fix floating punctuation
    text = FLOATING_PUNCTUATION.sub(r'\1 ', text)

    # Fix multiple dots
    text = MULTI_DOTS.sub(r'.', text)

    # Remove empty parentheses
    text = LONELY_PARENS.sub(r'', text)

    # Fix multiple dashes
    text = MULTIPLE_DASHES.sub(r'-', text)

    # Handle standalone equals signs (section separators)
    text = STANDALONE_EQUALS.sub(r' Section: ', text)

    # Fix standalone punctuation
    text = STANDALONE_SYMBOLS.sub(r'\1 ', text)

    return text


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

    # Performance optimization: Create a mapping of methods to functions
    method_mapping = {
        'camel': split_camel_case,
        'snake': split_snake_case,
        'kebab': split_kebab_case,
        'number': split_number_transitions,
        'team_names': split_team_names
    }

    result = text

    # Apply each method in sequence
    for method in methods:
        if method in method_mapping:
            result = method_mapping[method](result)

    # Clean up extra spaces and normalize
    return ' '.join(result.split())


def deep_text_cleaner(text):
    """
    Deep cleaner for problematic academic text like in the example.
    """
    if not isinstance(text, str):
        return text

    # Initial cleanup of excessive spaces
    result = ' '.join(text.split())

    # Clean academic references
    result = clean_academic_references(result)

    # Clean and normalize punctuation
    result = clean_punctuation(result)

    # Make sure first letter of sentences is capitalized
    sentences = re.split(r'(?<=[.!?])\s+', result)
    sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
    result = ' '.join(sentences)

    # Final cleanup of spacing
    result = ' '.join(result.split())

    return result


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

    # Apply general word splitting methods first
    result = general_word_splitter(text)

    # Apply deep cleaning for problematic text
    result = deep_text_cleaner(result)

    # Additional football-specific patterns

    # Handle parentheses spacing
    result = PARENTHESES_SPACING_START.sub(r'\1 (', result)
    result = PARENTHESES_SPACING_END.sub(r') \1', result)

    # Handle hyphenated dates/ranges - keep hyphen but ensure spaces around it
    result = HYPHENATED_DATES.sub(r'\1 - \2', result)

    # Remove unnecessary symbols and excessive spaces
    result = EXCESS_SPACES.sub(' ', result)  # Remove extra spaces
    # Remove spaces before punctuation
    result = SPACE_BEFORE_PUNCT.sub(r'\1', result)

    # Handle special football abbreviations
    result = SPECIAL_ABBREVIATIONS.sub('49ers', result)

    # Preserve special team names with numbers
    result = PRESERVE_NUMERIC_TEAMS.sub(r'\1', result)

    # Handle special characters like asterisks
    result = SPECIAL_CHARS.sub(
        lambda m: f"{m.group(1) or m.group(3)} {m.group(2) or m.group(4)}", result)

    # Standardize year ranges
    result = YEAR_RANGES.sub(r'(\1-\2)', result)

    # Clean up extra spaces and normalize
    return ' '.join(result.split())


def fix_messy_text(sample_text):
    """
    Example function to demonstrate fixing severely messy text like the provided example.
    """
    # First apply the general text cleaning
    cleaned = deep_text_cleaner(sample_text)

    # Then apply the football-specific cleaning
    cleaned = football_text_cleaner(cleaned)

    return cleaned
