from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import time
import random
import logging
from queue import Queue, Empty
from threading import Thread, Lock
import undetected_chromedriver as uc
import json
import os
import csv
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import argparse

# Global lock for thread-safe operations
lock = Lock()

# Configuration - these are now local constants, not globals
LOG_FILE = 'keyword_scraper.log'
MAX_THREADS = 5  # Number of concurrent scraper threads
BASE_TIMEOUT = 10  # Seconds
SCROLL_PAUSE_TIME = 0.5  # Time to wait after each scroll
MAX_SCROLLS = 5  # Maximum number of scrolls
MAX_RETRIES = 3  # Maximum retries for failing requests
REQUEST_DELAY = 1  # Delay between requests to avoid overloading the servers
DATA_DIRECTORY = "keyword_data"  # Directory for storing the data

# User Agent list
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/117.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0"
]

# Configure Logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class KeywordScraper:
    def __init__(self, base_url, keywords, file_format="csv", max_pages_per_keyword=10, driver=None):
        """Initialize the scraper with a base URL and keywords to search for.

        Args:
            base_url (str): The website to scrape
            keywords (list): List of keywords to search for
            file_format (str): Format to save data ('csv' or 'json')
            max_pages_per_keyword (int): Maximum pages to scrape per keyword
            driver: The Selenium driver instance. If None, it will be set up.
        """
        self.driver = driver  # Use provided driver
        self.base_url = base_url
        self.keywords = keywords
        self.visited = set()  # Track visited URLs to avoid duplicates
        self.file_format = file_format
        self.max_pages_per_keyword = max_pages_per_keyword

        # Extract domain name for file naming
        self.domain = urlparse(base_url).netloc.split('.')[1] if len(
            urlparse(base_url).netloc.split('.')) > 1 else urlparse(base_url).netloc

        # Create the data directory structure
        os.makedirs(DATA_DIRECTORY, exist_ok=True)

        # Dictionary to store all scraped data by keyword
        self.keyword_data = {keyword: [] for keyword in keywords}

    def setup_driver(self, headless=True):
        """Set up the Chrome WebDriver with anti-detection measures.

        Args:
            headless (bool): Whether to run Chrome in headless mode
        """
        chrome_options = Options()

        with lock:
            if headless:
                chrome_options.add_argument("--headless")
                chrome_options.add_argument('--disable-gpu')
                chrome_options.add_argument("--window-size=1920,1080")
                chrome_options.add_argument('--disable-dev-shm-usage')
                chrome_options.add_argument('--no-sandbox')

            # Anti-bot detection measures
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            user_agent = random.choice(USER_AGENTS)
            chrome_options.add_argument(f"--user-agent={user_agent}")

            # Additional privacy options
            chrome_options.add_argument("--disable-infobars")
            chrome_options.add_argument("--disable-extensions")

        try:
            self.driver = uc.Chrome(options=chrome_options)  # Using undetected chrome
            logging.info(f"WebDriver set up successfully for {self.base_url}")
        except WebDriverException as e:
            logging.error(f"Error setting up WebDriver: {e}")
            raise

    def close_driver(self):
        """Safely close the WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
                logging.info("WebDriver closed successfully")
            except Exception as e:
                logging.error(f"Error closing WebDriver: {e}")

    def scroll_page(self, timeout=BASE_TIMEOUT, max_scrolls=MAX_SCROLLS):
        """Scrolls the page to load lazy-loaded content.

        Args:
            timeout (int): Maximum wait time in seconds
            max_scrolls (int): Maximum number of scroll operations
        """
        try:
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            scrolls = 0

            while scrolls < max_scrolls:
                # Scroll down to the bottom
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                # Wait for new content to load
                time.sleep(SCROLL_PAUSE_TIME)

                # Calculate new scroll height and compare with last scroll height
                new_height = self.driver.execute_script("return document.body.scrollHeight")

                if new_height == last_height:
                    # If heights are the same, no new content loaded
                    break

                last_height = new_height
                scrolls += 1

                # Add some randomization to mimic human behavior
                if random.random() > 0.7:  # 30% chance to pause
                    time.sleep(random.uniform(0.5, 1.5))

            logging.info(f"Scrolled {scrolls} times on page")

        except Exception as e:
            logging.error(f"Error scrolling the page: {e}")

    def get_page_content(self, url, timeout=BASE_TIMEOUT, retries=MAX_RETRIES):
        """Loads the page and handles timeouts, with retries.

        Args:
            url (str): URL to retrieve
            timeout (int): Wait timeout in seconds
            retries (int): Number of retry attempts

        Returns:
            str or None: Page HTML source if successful, None otherwise
        """
        for attempt in range(retries):
            try:
                # Load the page
                self.driver.get(url)

                # Wait for page to load
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, 'body'))
                )

                # Scroll to load additional content
                self.scroll_page()

                # Get the page source
                return self.driver.page_source

            except TimeoutException:
                logging.warning(f"Timeout on {url}, attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    logging.error(f"Failed to load {url} after {retries} attempts.")
                    return None

            except Exception as e:
                logging.error(f"Error while loading {url} (attempt {attempt + 1}/{retries}): {e}")

                if attempt == retries - 1:
                    return None

            # Add delay between retries
            time.sleep(REQUEST_DELAY)

    def extract_links(self, html_source):
        """Extracts all internal links from a webpage.

        Args:
            html_source (str): HTML content of the page

        Returns:
            set: Set of absolute URLs found on the page
        """
        soup = BeautifulSoup(html_source, "html.parser")
        links = set()

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]

            # Convert relative URLs to absolute
            absolute_url = urljoin(self.base_url, href)

            # Only include internal links from the same domain
            if urlparse(absolute_url).netloc == urlparse(self.base_url).netloc:
                links.add(absolute_url)

        logging.info(f"Extracted {len(links)} internal links")
        return links

    def extract_data_with_keywords(self, html_source, selectors, keyword):
        """Extracts data from the page that contains the specified keyword.

        Args:
            html_source (str): HTML content of the page
            selectors (list): CSS selectors to use for extraction
            keyword (str): Keyword to search for in the content

        Returns:
            list: List of dictionaries containing matched content and metadata
        """
        try:
            soup = BeautifulSoup(html_source, 'html.parser')
            results = []

            # Current timestamp for metadata
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_url = self.driver.current_url

            # Extract the page title
            title = soup.title.string if soup.title else "No title"

            # Search through each selector
            for selector in selectors:
                for element in soup.select(selector):
                    text = element.get_text(strip=True)

                    # Check if the keyword is in the text (case insensitive)
                    if keyword.lower() in text.lower():
                        # Create a data item with metadata
                        data_item = {
                            "text": text,
                            "selector": selector,
                            "url": current_url,
                            "timestamp": timestamp,
                            "title": title,
                            "keyword": keyword
                        }

                        # Try to get the context (parent element's text)
                        if element.parent:
                            context = element.parent.get_text(strip=True)
                            if len(context) > len(text):
                                data_item["context"] = context

                        results.append(data_item)

            logging.info(f"Found {len(results)} items containing keyword '{keyword}'")
            return results

        except Exception as e:
            logging.error(f"Error extracting data with keyword '{keyword}': {e}")
            return []

    def save_keyword_data(self, keyword):
        """Saves all data for a specific keyword to a file.

        Args:
            keyword (str): The keyword to save data for
        """
        # Create a safe filename from the keyword and domain
        safe_keyword = keyword.replace(' ', '_').lower()
        filename = f"{safe_keyword}_data_{self.domain}"

        if self.file_format == 'json':
            self._save_json(keyword, filename)
        elif self.file_format == 'csv':
            self._save_csv(keyword, filename)
        else:
            logging.error(
                f"Invalid format to save file, it must be `json` or `csv`, not {self.file_format}")

    def _save_json(self, keyword, filename):
        """Saves keyword data to JSON file.

        Args:
            keyword (str): The keyword being saved
            filename (str): Base filename without extension
        """
        try:
            # Ensure directory exists
            os.makedirs(DATA_DIRECTORY, exist_ok=True)

            filepath = os.path.join(DATA_DIRECTORY, f"{filename}.json")

            # Add metadata
            data_to_save = {
                "keyword": keyword,
                "website": self.base_url,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "items_count": len(self.keyword_data[keyword]),
                "data": self.keyword_data[keyword]
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=4, ensure_ascii=False)

            logging.info(
                f"Saved {len(self.keyword_data[keyword])} items for keyword '{keyword}' to {filepath}")

        except Exception as e:
            logging.error(f"Error saving data to JSON file for keyword '{keyword}': {e}")

    def _save_csv(self, keyword, filename):
        """Saves keyword data to CSV file.

        Args:
            keyword (str): The keyword being saved
            filename (str): Base filename without extension
        """
        try:
            # Ensure directory exists
            os.makedirs(DATA_DIRECTORY, exist_ok=True)

            filepath = os.path.join(DATA_DIRECTORY, f"{filename}.csv")

            # If no data, create an empty file with headers
            if not self.keyword_data[keyword]:
                with open(filepath, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        ["keyword", "url", "title", "selector", "text", "timestamp", "context"])
                logging.info(f"Created empty CSV file for keyword '{keyword}' at {filepath}")
                return

            # Get all possible keys from all data items
            all_keys = set()
            for item in self.keyword_data[keyword]:
                all_keys.update(item.keys())

            # Ensure key order with most important fields first
            fieldnames = ["keyword", "url", "title", "selector", "text", "timestamp"]
            # Add any additional fields
            for key in all_keys:
                if key not in fieldnames:
                    fieldnames.append(key)

            with open(filepath, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.keyword_data[keyword])

            logging.info(
                f"Saved {len(self.keyword_data[keyword])} items for keyword '{keyword}' to {filepath}")

        except Exception as e:
            logging.error(f"Error saving data to CSV file for keyword '{keyword}': {e}")

    def search_keyword(self, keyword, selectors, queue=None):
        """Search for content related to a keyword.

        Args:
            keyword (str): Keyword to search for
            selectors (list): CSS selectors to search within
            queue (Queue, optional): Queue for breadth-first crawling

        Returns:
            int: Number of pages processed
        """
        pages_processed = 0
        links_to_visit = [self.base_url]  # Start with the base URL

        logging.info(f"Starting search for keyword '{keyword}' on {self.base_url}")

        try:
            while links_to_visit and pages_processed < self.max_pages_per_keyword:
                current_url = links_to_visit.pop(0)

                # Skip if already visited
                if current_url in self.visited:
                    continue

                self.visited.add(current_url)
                pages_processed += 1

                # Get page content
                html = self.get_page_content(current_url)
                if not html:
                    continue

                # Extract data for this keyword
                data = self.extract_data_with_keywords(html, selectors, keyword)

                # Add the data to our collection
                if data:
                    self.keyword_data[keyword].extend(data)

                # Find more links to visit if we need more pages
                if pages_processed < self.max_pages_per_keyword:
                    new_links = self.extract_links(html)

                    # Add new unvisited links
                    for link in new_links:
                        if link not in self.visited and link not in links_to_visit:
                            links_to_visit.append(link)

                # Add a delay between page loads
                time.sleep(REQUEST_DELAY + random.uniform(0.5, 1.5))

            logging.info(f"Completed search for '{keyword}' on {self.base_url}. "
                         f"Processed {pages_processed} pages, found {len(self.keyword_data[keyword])} items.")

            # Save the data for this keyword
            self.save_keyword_data(keyword)

            return pages_processed

        except Exception as e:
            logging.error(f"Error searching for keyword '{keyword}': {e}")
            return pages_processed

        finally:
            # If using a worker thread system, put any remaining links in the queue
            if queue and links_to_visit and pages_processed < self.max_pages_per_keyword:
                for link in links_to_visit:
                    if link not in self.visited:
                        queue.put((link, selectors, keyword))

    def run(self):
        """Run the scraper for all keywords."""
        try:
            # Define the selectors - these can be customized
            selectors = ["h1", "h2", "h3", "p", "li", "div.content", "article", "section"]

            for keyword in self.keywords:
                logging.info(f"Starting scraping for keyword: {keyword}")
                self.search_keyword(keyword, selectors)

            # Show summary
            for keyword in self.keywords:
                logging.info(f"Total items found for '{keyword}': {len(self.keyword_data[keyword])}")

        except Exception as e:
            logging.error(f"Error running scraper: {e}")

        finally:
            self.close_driver()


def worker(shared_data, queue, website, file_format, max_pages, selectors, driver):
    """Worker function for multi-threaded scraping.
    Args:
        shared_data (dict): Shared dictionary to store results
        queue (Queue): Queue of URLs to process
        website (str): Base website URL
        file_format (str): Format to save data ('csv' or 'json')
        max_pages (int): Maximum pages per keyword
        selectors (list): CSS selectors to use
        driver: Selenium driver instance to use

    """

    local_data = {}
    visited = set()

    try:
        while True:
            try:
                keyword, url = queue.get(timeout=5)

                # Initialize keyword data if needed
                if keyword not in local_data:
                    local_data[keyword] = []

                # Skip if already visited
                if url in visited:
                    queue.task_done()
                    continue

                # Create a local scraper instance using the shared driver
                scraper = KeywordScraper(website, [keyword], file_format, max_pages, driver)

                # Get page content
                html = scraper.get_page_content(url)
                if html:
                    # Extract data
                    data = scraper.extract_data_with_keywords(html, selectors, keyword)
                    if data:
                        with lock:
                            if keyword not in shared_data:
                                shared_data[keyword] = []
                            shared_data[keyword].extend(data)

                    # Find new links and add to the queue
                    new_links = scraper.extract_links(html)
                    for link in new_links:
                        if link not in visited:
                            with lock:  # Ensure thread-safe access to the queue
                                queue.put((keyword, link))

                visited.add(url)
                queue.task_done()

            except Empty:
                break

            except Exception as e:
                logging.error(f"Worker error: {e}")
                queue.task_done()

    finally:
        pass  # No need to close driver here



def setup_driver(headless=True):
    """Set up the Chrome WebDriver with anti-detection measures.

    Args:
        headless (bool): Whether to run Chrome in headless mode
    """
    chrome_options = Options()

    if headless:
        chrome_options.add_argument("--headless")
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--no-sandbox')

    # Anti-bot detection measures
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    user_agent = random.choice(USER_AGENTS)
    chrome_options.add_argument(f"--user-agent={user_agent}")

    # Additional privacy options
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-extensions")

    try:
        driver = uc.Chrome(options=chrome_options)  # Using undetected chrome
        logging.info(f"WebDriver set up successfully")
        return driver
    except WebDriverException as e:
        logging.error(f"Error setting up WebDriver: {e}")
        raise


def multi_threaded_keyword_search(driver, website, keywords, selectors, file_format, max_pages, max_threads=MAX_THREADS):
    """Run keyword search with multiple threads.

    Args:
        driver: Selenium driver instance
        website (str): Base URL to scrape
        keywords (list): Keywords to search for
        selectors (list): CSS selectors to use
        file_format (str): Format to save data ('csv' or 'json')
        max_pages (int): Maximum pages per keyword
        max_threads (int): Maximum number of worker threads
    """

    queue = Queue()
    shared_data = {}

    # Initialize the queue with all keywords and the base URL
    for keyword in keywords:
        queue.put((keyword, website))
        shared_data[keyword] = []  # Initialize shared data for each keyword

    # Create and start worker threads, passing in the shared driver
    threads = []
    for _ in range(max_threads):
        thread = Thread(target=worker, args=(shared_data, queue, website, file_format, max_pages, selectors, driver))
        thread.daemon = True
        thread.start()
        threads.append(thread)

    # Wait for all tasks to complete
    queue.join()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # After threads complete, save the data
    scraper = KeywordScraper(website, keywords, file_format, max_pages, driver=driver) # Pass driver

    for keyword in keywords:
        scraper.keyword_data[keyword] = shared_data[keyword]  # Get data from shared data
        scraper.save_keyword_data(keyword)


def main():
    """Main function to run the scraper."""
    parser = argparse.ArgumentParser(description='Keyword-based web scraper')
    parser.add_argument('--website', type=str, required=True, help='Website URL to scrape')
    parser.add_argument('--keywords', type=str, nargs='+', required=True, help='Keywords to search for')
    parser.add_argument('--threads', type=int, default=2, help='Number of threads to use')  # Reduced threads
    parser.add_argument('--format', type=str, choices=['json', 'csv'], default='csv',
                        help='Output file format')
    parser.add_argument('--max-pages', type=int, default=10,
                        help='Maximum pages to scrape per keyword')

    args = parser.parse_args()

    logging.info(f"Starting keyword scraper for website: {args.website}")
    logging.info(f"Keywords to search: {args.keywords}")

    # Define selectors - these can be customized based on the website structure
    selectors = ["h1", "h2", "h3", "p", "li", "div.content", "article", "section"]

    # Initialize undetected chromedriver here - initialize the driver ONCE!
    driver = setup_driver()
    try:
        multi_threaded_keyword_search(driver, args.website, args.keywords, selectors, args.format, args.max_pages, args.threads)
    finally:
        driver.quit()  # Ensure driver is closed in all cases
    logging.info("Scraping complete!")


if __name__ == '__main__':
    main()