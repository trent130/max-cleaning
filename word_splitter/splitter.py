import re
from typing import Optional, List, Dict, Callable, Any

# Precompiled regex patterns for performance
class RegexPatterns:
    CAMEL_CASE = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')
    NUMBER_TRANSITION = re.compile(r'(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)')
    TEAM_NAME_THE = re.compile(r'the([A-Z][a-z]+)')
    ADJACENT_CAPS = re.compile(r'([A-Z][a-z]+)([A-Z][a-z]+)')
    PARENTHESES_SPACING_START = re.compile(r'([A-Za-z])\(')
    PARENTHESES_SPACING_END = re.compile(r'\)([A-Za-z])')
    HYPHENATED_DATES = re.compile(r'(\d+)-(\d+)')
    EXCESS_SPACES = re.compile(r'\s+')
    SPACE_BEFORE_PUNCT = re.compile(r'(\s)[,.!?]')
    SPECIAL_ABBREVIATIONS = re.compile(r'\b(49 ers)\b')
    PRESERVE_NUMERIC_TEAMS = re.compile(r'(\d+ers|\d+ERS)')
    # SPECIAL_CHARS = re.compile(r'([*&])([A-Za-z])|([A-Za-z])([*&])')
    # YEAR_RANGES = re.compile(r'\((\d{4})\s*–\s*(\d{4})\)')
    FLOATING_PUNCTUATION = re.compile(r'\s*([.,;:?!])\s*')
    BRACKETS_WITH_NUMBERS = re.compile(r'\[\s*(\d+)\s*\]\s*:\s*(\d+)')
    BRACKETS_SIMPLE = re.compile(r'\[\s*(\d+)\s*\]')
    #QUOTES_SPACING = re.compile(r'``\s*(.*?)\s*\'\'')
    MULTI_DOTS = re.compile(r'\.{2,}')
    LONELY_PARENS = re.compile(r'\(\s*\)')
    MULTIPLE_DASHES = re.compile(r'-{2,}')
    MULTIPLE_PUNCTUATIONS = re.compile(r"`{1,}")
    STANDALONE_EQUALS = re.compile(r'\s*=+\s*')
    STANDALONE_SYMBOLS = re.compile(r'\s+([.,;:?!])\s+')
    BACKTICKS = re.compile(r'\s*`+\s*')
    CARET = re.compile(r'\s*\^+\s*')
    SEE_ALSO_SECTION = re.compile(r'===+\s*see also\s*===+', re.IGNORECASE)
    REFERENCE_SECTION = re.compile(r'===+\s*reference\s*===+', re.IGNORECASE)
    PUNCTUATIONS = re.compile(r"[^\w\s$.,-]")
    CURRENCY_STANDARDIZER = re.compile(r'\$\s*([\d\s.]+)')
    REDUNDANT_QUOTES = re.compile(r'\'\'\s*')

def _ensure_string(text: Any) -> Optional[str]:
    """Ensure input is a string; return None if not."""
    text = re.sub(r"``", '', text)
    return text if isinstance(text, str) else None

def split_camel_case(text: Any) -> Optional[str]:
    """Split camelCase or PascalCase text into separate words.

    Args:
        text: Input text to process.

    Returns:
        Processed text with spaces, or original input if not a string.
    """
    text = _ensure_string(text)
    return RegexPatterns.CAMEL_CASE.sub(' ', text) if text else text

def split_snake_case(text: Any) -> Optional[str]:
    """Split snake_case text into separate words."""
    text = _ensure_string(text)
    return text.replace('_', ' ') if text else text

def split_kebab_case(text: Any) -> Optional[str]:
    """Split kebab-case text into separate words."""
    text = _ensure_string(text)
    return text.replace('-', ' ') if text else text

def split_number_transitions(text: Any) -> Optional[str]:
    """Split transitions between numbers and letters, preserving special cases."""
    text = _ensure_string(text)
    if not text:
        return text

    # Protected patterns (e.g., team names, common identifiers)
    protected_patterns = {'49ers', '76ers', '3M', '7Eleven', '5G', '4K', '3D'}
    replacements: Dict[str, str] = {}
    working_text = text

    # Temporarily replace protected patterns
    for i, pattern in enumerate(protected_patterns):
        placeholder = f"__PROTECTED_{i}__"
        if pattern in working_text:
            replacements[placeholder] = pattern
            working_text = working_text.replace(pattern, placeholder)

    # Apply number splitting
    working_text = RegexPatterns.NUMBER_TRANSITION.sub('', working_text)

    # Restore protected patterns
    for placeholder, original in replacements.items():
        working_text = working_text.replace(placeholder, original)

    return working_text

def split_team_names(text: Any) -> Optional[str]:
    """Split patterns commonly found in team names."""
    text = _ensure_string(text)
    if not text:
        return text
    result = RegexPatterns.TEAM_NAME_THE.sub(r'the \1', text)
    return RegexPatterns.ADJACENT_CAPS.sub(r'\1 \2', result)

def clean_academic_references(text: Any) -> Optional[str]:
    """Clean academic reference patterns like '[12]: 36' or '[2]'."""
    text = _ensure_string(text)
    if not text:
        return text
    text = RegexPatterns.BRACKETS_WITH_NUMBERS.sub(r' (Reference \1, p.\2) ', text)
    return RegexPatterns.BRACKETS_SIMPLE.sub(r' (Reference \1) ', text)

def clean_punctuation(text: Any) -> Optional[str]:
    """Clean and normalize punctuation in text."""
    text = _ensure_string(text)
    if not text:
        return text

    text = re.sub('``', '', text)
    
    text = RegexPatterns.SEE_ALSO_SECTION.sub(' See also ', text)
    text = RegexPatterns.REFERENCE_SECTION.sub(' Reference ', text)
    text = RegexPatterns.BACKTICKS.sub('', text)  
    text = RegexPatterns.CARET.sub('', text)     
    #text = RegexPatterns.QUOTES_SPACING.sub(r'"\1"', text)
    text = RegexPatterns.FLOATING_PUNCTUATION.sub(r'\1 ', text)
    text = RegexPatterns.MULTI_DOTS.sub(r'.', text)
    text = RegexPatterns.LONELY_PARENS.sub(r'', text)
    text = RegexPatterns.MULTIPLE_DASHES.sub(r'-', text)
    text = RegexPatterns.STANDALONE_EQUALS.sub(r'', text)
    text = RegexPatterns.STANDALONE_SYMBOLS.sub(r'\1 ', text)
    text = RegexPatterns.PUNCTUATIONS.sub(r"'", text)
    text = RegexPatterns.MULTIPLE_PUNCTUATIONS.sub('', text)
    text = RegexPatterns.CURRENCY_STANDARDIZER.sub(" dollar ", text)
    text = RegexPatterns.REDUNDANT_QUOTES.sub("", text)
    text = RegexPatterns.EXCESS_SPACES.sub(' ', text)
    return text

def general_word_splitter(text: Any, methods: Optional[List[str]] = None) -> Optional[str]:
    """General-purpose function to split concatenated words using multiple methods.

    Args:
        text: The text to process.
        methods: List of splitting methods to apply. Options: 'camel', 'snake', 'kebab',
            'number', 'team_names'. If None, applies all.

    Returns:
        Processed text with words split, or original input if not a string.
    """
    text = _ensure_string(text)
    if not text:
        return text
    
    # text = re.sub('===', '', text)
    default_methods = ['camel', 'snake', 'kebab', 'number', 'team_names']
    methods = methods if methods is not None else default_methods

    method_mapping: Dict[str, Callable[[str], Optional[str]]] = {
        'camel': split_camel_case,
        'snake': split_snake_case,
        'kebab': split_kebab_case,
        'number': split_number_transitions,
        'team_names': split_team_names
    }

    result = text
    result = clean_punctuation(result)
    for method in methods:
        if method in method_mapping:
            result = method_mapping[method](result) or result
    return RegexPatterns.EXCESS_SPACES.sub(' ', result).strip()
    
def deep_text_cleaner(text: Any) -> Optional[str]:
    """Deep cleaner for problematic academic text.

    Args:
        text: Text to clean.

    Returns:
        Cleaned text with proper formatting, or original input if not a string.
    """
    text = _ensure_string(text)
    if not text:
        return text

    result = RegexPatterns.EXCESS_SPACES.sub(' ', text).strip()
    result = clean_academic_references(result)
    result = clean_punctuation(result)

    # Capitalize first letter of sentences
    sentences = re.split(r'(?<=[.!?])\s+', result)
    result = ' '.join(s[0].upper() + s[1:] if s else s for s in sentences)
    return RegexPatterns.EXCESS_SPACES.sub(' ', result).strip()

def football_text_cleaner(text: Any) -> Optional[str]:
    """Specialized cleaner for football-related text with specific patterns.

    Args:
        text: Football text to clean.

    Returns:
        Cleaned and normalized football text.
    """
    text = _ensure_string(text)
    if not text:
        return text

    result = general_word_splitter(text)
    result = deep_text_cleaner(result)

    result = RegexPatterns.PARENTHESES_SPACING_START.sub(r'\1 (', result)
    result = RegexPatterns.PARENTHESES_SPACING_END.sub(r') \1', result)
    result = RegexPatterns.HYPHENATED_DATES.sub(r'\1 - \2', result)
    result = RegexPatterns.EXCESS_SPACES.sub(' ', result)
    result = RegexPatterns.SPACE_BEFORE_PUNCT.sub(r'\1', result)
    result = RegexPatterns.SPECIAL_ABBREVIATIONS.sub('49ers', result)
    result = RegexPatterns.PRESERVE_NUMERIC_TEAMS.sub(r'\1', result)
    result = RegexPatterns.SPECIAL_CHARS.sub(
        lambda m: f"{m.group(1) or m.group(3)} {m.group(2) or m.group(4)}", result)
    result = RegexPatterns.YEAR_RANGES.sub(r'(\1-\2)', result)
    return result.strip()

def fix_messy_text(sample_text: Any) -> Optional[str]:
    """Demonstrate fixing severely messy text.

    Args:
        sample_text: Messy text to clean.

    Returns:
        Fully cleaned text.
    """
    cleaned = deep_text_cleaner(sample_text)
    return football_text_cleaner(cleaned) if cleaned else cleaned
