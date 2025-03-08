# MaxCleaner: A Robust Data Preprocessing Toolkit

**TextCleaner** is a comprehensive Python library designed to streamline and enhance the data preprocessing stage for text-based datasets. It provides a suite of powerful tools for cleaning, standardizing, and enriching textual data, enabling you to build more accurate and reliable machine learning models.


## Table of Contents

- [Introduction](#introduction)
- [Features](#key-features)
- [Installation](#installation)
- [why use it](#why-use-textcleaner)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction
This library is designed with flexibility and efficiency in mind, offering various processing strategies (sequential, multiprocessing, and vectorized) to cater to datasets of varying sizes and computational resources.  It intelligently handles common data issues, provides robust URL and datetime standardization, and integrates seamlessly with the popular pandas DataFrame structure.

## Key Features

*   **Text Cleaning:**
    *   Removes noise, handles whitespace irregularities, and normalizes text.
    *   Offers flexible word splitting using multiple strategies, including camelCase, snake\_case, and kebab-case.
    *   Provides specialized cleaning for specific text types (e.g., football datasets).

*   **URL Standardization:**
    *   Standardizes URLs, removes unnecessary query parameters (UTM tracking, etc.), normalizes case, and handles trailing slashes.
    *   Extracts links from HTML content and identifies potential broken links.
    *   Caches URL parsing for increased efficiency.

*   **Datetime Processing:**
    *   Detects and standardizes datetime formats within text and in DataFrame columns.
    *   Provides configurable output formats and robust error handling.

*   **Multilingual Support:**
    *   Supports multiple languages through integration with the NLTK library.

*   **Scalable Processing:**
    *   Offers sequential, multiprocessing, and vectorized processing strategies for optimal performance.
    *   Memory monitoring to prevent out-of-memory errors.

*   **Pandas Integration:**
    *   Seamlessly integrates with pandas DataFrames, making it easy to process entire datasets with a single function call.

*   **NLTK Integration (Optional):**
    *   Leverages NLTK for advanced text processing features such as tokenization, stop word removal, and lemmatization (if NLTK is installed).

*   **Configurable:**
    *   Highly configurable through a `CleanerConfig` dataclass or you can extend the configs into a file(`config.yaml`), allowing you to tailor the cleaning process to your specific needs.


## Installation

To get started with TextCleaner, clone the repository and install the required dependencies.

```bash
git clone https://github.com/trent130/max-cleaning.git
cd max-cleaning
pip install -r requirements.txt
```


## Why Use TextCleaner?

*   **Save Time:** Automate tedious and time-consuming data cleaning tasks.
*   **Improve Data Quality:** Enhance the consistency and accuracy of your text data.
*   **Boost Model Performance:** Build better-performing machine learning models with clean and standardized data.
*   **Increase Scalability:** Process large datasets efficiently using multiprocessing and vectorized operations.
*   **Reduce Errors:** Minimize the risk of errors and inconsistencies in your data preprocessing pipeline.

## Usage

Max Cleaning can be used through the command line or integrated into your Python projects.

### Command Line

To run Max Cleaning from the command line, use the following command:

```bash
python main.py --config config.yaml
```

### Python Integration

You can also integrate Max Cleaning into your Python scripts:

```python
from max_cleaning import MaxCleaning

config = "config.yaml"
cleaner = MaxCleaning(config)
cleaner.run()
```

## Configuration

Max Cleaning uses a `CleanerConfig` dataclass mostly but you can use a configuration file (`config.yaml`) to define cleaning rules and settings. Below is an example configuration:

```yaml
data_cleaning:
  remove_duplicates: true
  handle_missing_values: true
  format_rules:
    - column: date
      format: "%Y-%m-%d"

file_organization:
  sort_by: "date"
  destination: "/organized_files"
```

## Contributing

I welcome contributions to TextCleaner! If you have an idea for a new feature or have found a bug, please open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact me (the repository owner) at [trent130](https://github.com/trent130).