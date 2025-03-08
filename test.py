import time
import random
import logging
from queue import Queue, Empty
from threading import Thread, Lock
import undetected_chromedriver as uc
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import json
import os
import csv
from datetime import datetime
import argparse
from functools import lru_cache
import threading

# Configuration constants
LOG_FILE = 'keyword_scraper.log'
MAX_THREADS = 1
BASE_TIMEOUT = 15  # Slightly reduced timeout
SCROLL_PAUSE_TIME = 0.1  # Faster scrolling
MAX_SCROLLS = 3  # Reduced scrolls
MAX_RETRIES = 2
REQUEST_DELAY = 0.2  # Significantly reduced base delay
DATA_DIRECTORY = "keyword_data"

# User Agent rotation for better anti-detection
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
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Create a lock to make file writing thread-safe
link_file_lock = threading.Lock()

# Global lock for thread-safe operations
lock = Lock()


class KeywordScraper:
    def __init__(self, base_url, keywords, file_format="csv", max_pages_per_keyword=10, driver=None):
        """Initialize the scraper with a base URL and keywords to search for."""
        self.driver = driver
        self.base_url = base_url
        self.keywords = keywords
        self.visited = set()
        self.file_format = file_format
        self.max_pages_per_keyword = max_pages_per_keyword

        # Extract domain for file naming (improved)
        parsed_url = urlparse(base_url)
        domain_parts = parsed_url.netloc.split('.')
        self.domain = domain_parts[-2] if len(domain_parts) > 1 else parsed_url.netloc

        # Create data directory
        os.makedirs(DATA_DIRECTORY, exist_ok=True)

        # Dictionary to store scraped data by keyword
        self.keyword_data = {keyword: [] for keyword in keywords}

    def setup_driver(self, headless=True):
        """Set up the Chrome WebDriver with anti-detection measures."""
        chrome_options = Options()

        if headless:
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--no-sandbox')

        # Anti-bot detection measures
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")

        # Performance and privacy options
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--disable-notifications")

        # Additional performance improvements
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-setuid-sandbox")
        chrome_options.add_argument("--disable-features=TranslateUI")
        chrome_options.add_argument("--disable-web-security")

        try:
            self.driver = uc.Chrome(options=chrome_options) # Let undetected_chromedriver manage
            #  self.driver = uc.Chrome(options=chrome_options, driver_executable_path="/path/to/chromedriver") # If manual management is needed
            logging.info(f"WebDriver set up successfully for {self.base_url}")
        except WebDriverException as e:
            logging.error(f"Error setting up WebDriver: {e}")
            raise

        return self.driver

    def close_driver(self):
        """Safely close the WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
                logging.info("WebDriver closed successfully")
            except Exception as e:
                logging.error(f"Error closing WebDriver: {e}")

    def scroll_page(self, timeout=BASE_TIMEOUT, max_scrolls=MAX_SCROLLS):
        """Scrolls the page to load lazy-loaded content. Uses dynamic timing."""
        try:
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            scrolls = 0

            # More direct scrolling - less random behavior for speed
            scroll_script = "window.scrollTo(0, document.body.scrollHeight);"

            while scrolls < max_scrolls:
                self.driver.execute_script(scroll_script)
                time.sleep(SCROLL_PAUSE_TIME) # Reduced pause

                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break # Break faster if no new content
                last_height = new_height
                scrolls += 1

            logging.info(f"Scrolled {scrolls} times on page")

        except Exception as e:
            logging.error(f"Error scrolling the page: {e}")

    def get_page_content(self, url, timeout=BASE_TIMEOUT, retries=MAX_RETRIES):
        for attempt in range(retries):
            try:
                # Load the page
                self.driver.get(url)

                # If driver seems to be in a bad state, recreate it
                if attempt > 0:
                    self.close_driver()
                    self.setup_driver(headless=True)

                # Wait for page to load - reduced timeout
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "body *"))
                )

                # No extra jitter for speed
                # if random.random() > 0.7:
                #     time.sleep(random.uniform(0.5, 1.2))

                # Get the page source
                return self.driver.page_source

            except TimeoutException:
                logging.warning(f"Timeout on {url}, attempt {attempt + 1}/{retries}")
                # Reduced timeout increase on retry
                timeout = timeout * 1.2

            except Exception as e:
                logging.error(f"Error loading {url} (attempt {attempt + 1}/{retries}): {e}")

            if attempt < retries - 1:
                # Reduced backoff
                wait_time = REQUEST_DELAY * (attempt + 1) # Less exponential backoff
                logging.info(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)

        logging.error(f"Failed to load {url} after {retries} attempts.")
        return None

    @lru_cache(maxsize=100)
    def get_domain(self, url):
        """Extract domain from URL with caching for performance."""
        return urlparse(url).netloc

    def extract_links(self, html_source):
        """Extracts all internal links from a webpage, prioritizing by relevance."""
        soup = BeautifulSoup(html_source, "html.parser")
        links = set()

        from urllib.parse import urljoin, urlparse
        links = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            absolute_url = urljoin(self.base_url, href)
            # Only keep links from the same domain
            if urlparse(absolute_url).netloc == urlparse(self.base_url).netloc:
                links.add(absolute_url)

        # Commenting out file writing for speed - can re-enable if needed for debugging
        # with link_file_lock:
        #     with open('extracted_links.txt', 'a') as f:
        #         for link in links:
        #             f.write(link + '\n')

        prioritized_links = []

        base_domain = self.get_domain(self.base_url)

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]

            # Skip empty links and javascript links
            if not href or href.startswith("javascript:") or href == "#":
                continue

            # Convert relative URLs to absolute
            absolute_url = urljoin(self.base_url, href)

            # Only include internal links from the same domain
            link_domain = self.get_domain(absolute_url)
            if link_domain == base_domain:
                # Check if link looks more relevant (contains more content)
                link_text = a_tag.get_text(strip=True)
                parent_text = ""
                if a_tag.parent:
                    parent_text = a_tag.parent.get_text(strip=True)

                # Prioritize links with more descriptive text - reduced priority
                priority = 0
                if len(link_text) > 8: # Slightly less strict
                    priority += 3 # Reduced priority
                if len(parent_text) > 40: # Slightly less strict
                    priority += 2 # Reduced priority

                # Prioritize links with fewer query parameters - same priority
                if "?" not in absolute_url:
                    priority += 2

                # Store with priority
                prioritized_links.append((absolute_url, priority))

        # Sort by priority (higher first) and add to final set
        prioritized_links.sort(key=lambda x: x[1], reverse=True)
        links = {link for link, _ in prioritized_links}

        logging.info(f"Extracted {len(links)} internal links")
        return links

    def extract_data_with_keywords(self, html_source, selectors, keyword):
        """Extracts data from the page that contains the specified keyword."""
        try:
            soup = BeautifulSoup(html_source, 'html.parser')
            results = []

            # Current timestamp for metadata
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_url = self.driver.current_url

            # Extract the page title
            title = soup.title.string if soup.title else "No title"

            # Create keyword variations for more robust matching
            keyword_variations = [
                keyword.lower(),
                keyword.upper(),
                keyword.capitalize(),
                ' ' + keyword.lower() + ' '  # Word boundary matching
            ]

            # Search through each selector
            for selector in selectors:
                for element in soup.select(selector):
                    text = element.get_text(strip=True)

                    # Check if any keyword variation is in the text
                    found_match = False
                    for variation in keyword_variations:
                        if variation in text:
                            found_match = True
                            break

                    if found_match:
                        # Create a data item with metadata
                        data_item = {
                            "text": text,
                            "selector": selector,
                            "url": current_url,
                            "timestamp": timestamp,
                            "title": title,
                            "keyword": keyword
                        }

                        # Get context - simplified context extraction for speed
                        if element.parent:
                            context = element.parent.get_text(strip=True)
                            if len(context) > len(text) and len(context) < 500: # Reduced context length
                                data_item["context"] = context


                        results.append(data_item)

            logging.info(f"Found {len(results)} items containing keyword '{keyword}'")
            return results

        except Exception as e:
            logging.error(f"Error extracting data with keyword '{keyword}': {e}")
            return []

    def save_keyword_data(self, keyword):
        """Saves all data for a specific keyword to a file."""
        # Create a safe filename from the keyword and domain
        safe_keyword = ''.join(c if c.isalnum() else '_' for c in keyword.lower())
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        filename = f"{safe_keyword}_{self.domain}_{timestamp}"

        if self.file_format == 'json':
            self._save_json(keyword, filename)
        elif self.file_format == 'csv':
            self._save_csv(keyword, filename)
        else:
            logging.error(f"Invalid format: {self.file_format}, defaulting to CSV")
            self._save_csv(keyword, filename)

    def _save_json(self, keyword, filename):
        """Saves keyword data to JSON file."""
        try:
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
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)

            logging.info(f"Saved {len(self.keyword_data[keyword])} items for keyword '{keyword}' to {filepath}")

        except Exception as e:
            logging.error(f"Error saving data to JSON file for keyword '{keyword}': {e}")

    def _save_csv(self, keyword, filename):
        """Saves keyword data to CSV file."""
        try:
            filepath = os.path.join(DATA_DIRECTORY, f"{filename}.csv")

            # If no data, create an empty file with headers
            if not self.keyword_data[keyword]:
                with open(filepath, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["keyword", "url", "title", "selector", "text", "timestamp", "context"])
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
                for item in self.keyword_data[keyword]:
                    # Ensure all fields exist in each row
                    row = {field: item.get(field, "") for field in fieldnames}
                    writer.writerow(row)

            logging.info(f"Saved {len(self.keyword_data[keyword])} items for keyword '{keyword}' to {filepath}")

        except Exception as e:
            logging.error(f"Error saving data to CSV file for keyword '{keyword}': {e}")

    def search_keyword(self, keyword, selectors, queue=None):
        """Search for content related to a keyword."""
        pages_processed = 0
        links_to_visit = [self.base_url]  # Start with the base URL

        logging.info(f"Starting search for keyword '{keyword}' on {self.base_url}")

        try:
            while links_to_visit and pages_processed < self.max_pages_per_keyword:
                current_url = links_to_visit.pop(0)

                # Skip if already visited
                if current_url in self.visited:
                    logging.info(f"Visitrd page: {current_url}")
                    continue

                self.visited.add(current_url)
                pages_processed += 1

                # Get page content
                html = self.get_page_content(current_url)
                if not html:
                    continue

                logging.info(f"Visiting page: {current_url}")
                # Extract data for this keyword
                data = self.extract_data_with_keywords(html, selectors, keyword)

                # Add the data to our collection
                if data:
                    with lock:
                        self.keyword_data[keyword].extend(data)

                # Find more links to visit if we need more pages
                if pages_processed < self.max_pages_per_keyword:
                    new_links = self.extract_links(html)

                    # Add new unvisited links
                    for link in new_links:
                        if link not in self.visited and link not in links_to_visit:
                            links_to_visit.append(link)

                # Reduced delay here as well
                time.sleep(REQUEST_DELAY)

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
            # Define the selectors
            selectors = [
                "h1", "h2", "h3", "h4", "p", "li",
                "div.content", "article", "section",
                "div.main", "div.article", "main"
            ]

            # More efficient with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(self.keywords), MAX_THREADS)) as executor:
                futures = {executor.submit(self.search_keyword, keyword, selectors): keyword
                           for keyword in self.keywords}

                for future in futures:
                    keyword = futures[future]
                    try:
                        pages = future.result()
                        logging.info(f"Processed {pages} pages for keyword '{keyword}'")
                    except Exception as e:
                        logging.error(f"Error processing keyword '{keyword}': {e}")

            # Show summary
            for keyword in self.keywords:
                logging.info(f"Total items found for '{keyword}': {len(self.keyword_data[keyword])}")

        except Exception as e:
            logging.error(f"Error running scraper: {e}")

        finally:
            # Don't close driver here if it was passed in
            if hasattr(self, 'driver') and self.driver and not hasattr(self, '_driver_external'):
                self.close_driver()


def worker(shared_data, queue, website, file_format, max_pages, selectors, driver_pool):
    """Worker function for multi-threaded scraping with driver pooling."""

    # Get a driver from the pool
    driver = driver_pool.get_driver()

    local_visited = set()

    try:
        while True:
            try:
                keyword, url = queue.get(timeout=5)

                # Skip if already visited
                if url in local_visited:
                    queue.task_done()
                    continue

                local_visited.add(url)

                # Get page content
                html = None
                try:
                    # Load the page
                    driver.get(url)

                    # Wait for page to load - reduced timeout in worker as well
                    WebDriverWait(driver, BASE_TIMEOUT).until(
                        EC.presence_of_element_located((By.TAG_NAME, 'body'))
                    )

                    # Scroll to load additional content - keep scrolling in worker if needed for dynamic content sites
                    last_height = driver.execute_script("return document.body.scrollHeight")
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(SCROLL_PAUSE_TIME)

                    # Get the page source
                    html = driver.page_source

                except Exception as e:
                    logging.error(f"Error loading page {url}: {e}")

                if html:
                    # Extract data
                    soup = BeautifulSoup(html, 'html.parser')
                    results = []

                    # Current timestamp and URL for metadata
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
                                    "url": url,
                                    "timestamp": timestamp,
                                    "title": title,
                                    "keyword": keyword
                                }

                                # Try to get the context - simplified in worker as well
                                if element.parent:
                                    context = element.parent.get_text(strip=True)
                                    if len(context) > len(text) and len(context) < 500:
                                        data_item["context"] = context

                                results.append(data_item)

                    # Add the data to shared collection
                    if results:
                        with lock:
                            if keyword not in shared_data:
                                shared_data[keyword] = []
                            shared_data[keyword].extend(results)

                    # Find new links and add to the queue
                    links = set()
                    for a_tag in soup.find_all("a", href=True):
                        href = a_tag["href"]

                        # Convert relative URLs to absolute
                        absolute_url = urljoin(website, href)

                        # Only include internal links from the same domain
                        if urlparse(absolute_url).netloc == urlparse(website).netloc:
                            links.add(absolute_url)

                    for link in links:
                        if link not in local_visited:
                            with lock:  # Ensure thread-safe access to the queue
                                queue.put((keyword, link))

                queue.task_done()

                # Reduced delay in worker
                time.sleep(REQUEST_DELAY)

            except Empty:
                break

            except Exception as e:
                logging.error(f"Worker error: {e}")
                queue.task_done()

    finally:
        # Return the driver to the pool
        driver_pool.return_driver(driver)


class DriverPool:
    """A pool of WebDriver instances for concurrent scraping."""

    def __init__(self, pool_size=MAX_THREADS, headless=True):
        """Initialize the driver pool."""
        self.pool_size = pool_size
        self.headless = headless
        self.drivers = []
        self.available_drivers = Queue()
        self.lock = Lock()

        # Initialize pool
        self._initialize_pool()

    def _initialize_pool(self):
        """Create the initial pool of drivers."""
        for _ in range(self.pool_size):
            driver = self._create_driver()
            self.drivers.append(driver)
            self.available_drivers.put(driver)

        logging.info(f"Initialized driver pool with {self.pool_size} drivers")

    def _create_driver(self):
        """Create a new WebDriver instance."""
        chrome_options = Options()

        # if self.headless:
        #     chrome_options.add_argument("--headless=new")
        #     chrome_options.add_argument('--disable-gpu')
        #     chrome_options.add_argument("--window-size=1920,1080")

        # Anti-bot detection
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")

        # # Performance options
        # chrome_options.add_argument("--disable-infobars")
        # chrome_options.add_argument("--disable-extensions")
        # chrome_options.add_argument("--disable-dev-shm-usage")
        # chrome_options.add_argument("--no-sandbox")

        try:
            driver = uc.Chrome(options=chrome_options)
            return driver
        except WebDriverException as e:
            logging.error(f"Error creating driver: {e}")
            raise

    def get_driver(self):
        """Get a driver from the pool."""
        try:
            return self.available_drivers.get(timeout=30)
        except Empty:
            with self.lock:
                # If no drivers available, create a new one - keep creating if exhausted, might indicate driver issues
                logging.warning("Driver pool exhausted, creating new driver")
                driver = self._create_driver()
                self.drivers.append(driver)
                return driver

    def return_driver(self, driver):
        """Return a driver to the pool."""
        self.available_drivers.put(driver)

    def close_all(self):
        """Close all drivers in the pool."""
        for driver in self.drivers:
            try:
                driver.quit()
            except Exception as e:
                logging.error(f"Error closing driver: {e}")

        logging.info(f"Closed all {len(self.drivers)} drivers in the pool")


def multi_threaded_keyword_search(website, keywords, selectors, file_format, max_pages, max_threads=MAX_THREADS):
    """Run keyword search with multiple threads using a driver pool."""

    queue = Queue()
    shared_data = {}

    # Create a driver pool
    driver_pool = DriverPool(pool_size=max_threads)

    try:
        # Initialize the queue with all keywords and the base URL
        for keyword in keywords:
            queue.put((keyword, website))
            shared_data[keyword] = []  # Initialize shared data for each keyword

        # Create and start worker threads
        threads = []
        for _ in range(max_threads):
            thread = Thread(target=worker, args=(shared_data, queue, website, file_format, max_pages, selectors, driver_pool))
            thread.daemon = True
            thread.start()
            threads.append(thread)

        # Wait for queue to empty
        queue.join()

        # Wait for all threads to finish
        for thread in threads:
            thread.join(timeout=1)

        # Save the data
        driver = driver_pool.get_driver()
        scraper = KeywordScraper(website, keywords, file_format, max_pages, driver=driver)
        scraper._driver_external = True  # Mark driver as external

        for keyword in keywords:
            scraper.keyword_data[keyword] = shared_data[keyword]
            scraper.save_keyword_data(keyword)

        driver_pool.return_driver(driver)

    finally:
        # Clean up resources
        driver_pool.close_all()


def main():
    """Main function to run the scraper."""
    parser = argparse.ArgumentParser(description='Keyword-based web scraper')
    parser.add_argument('--website', type=str, required=True, help='Website URL to scrape')
    parser.add_argument('--keywords', type=str, nargs='+', required=True, help='Keywords to search for')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to use') # Default to 1 thread for initial test
    parser.add_argument('--format', type=str, choices=['json', 'csv'], default='csv',
                        help='Output file format')
    parser.add_argument('--max-pages', type=int, default=10, # Reduced default max pages for faster testing
                        help='Maximum pages to scrape per keyword')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='Run in headless mode')
    parser.add_argument('--selectors', type=str, nargs='+',
                        default=["h1", "h2", "h3", "p", "li", "div.content", "article", "section"],
                        help='CSS selectors to search for keywords')

    args = parser.parse_args()

    # Configure logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=log_format)

    # Add console handler for better debugging
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger('').addHandler(console)

    logging.info(f"Starting keyword scraper for website: {args.website}")
    logging.info(f"Keywords to search: {args.keywords}")

    try:
        multi_threaded_keyword_search(
            args.website,
            args.keywords,
            args.selectors,
            args.format,
            args.max_pages,
            args.threads
        )
        logging.info("Scraping complete!")
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())