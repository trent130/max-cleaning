from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, SessionNotCreatedException
import time
import random
import logging
from queue import Queue, Empty
from threading import Thread, Lock, Event
import undetected_chromedriver as uc
import json
import os
import csv
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import threading
import requests
import backoff
import socket
import subprocess
import signal
import atexit

# Configuration constants
LOG_FILE = 'keyword_scraper.log'
MAX_THREADS = 1
BASE_TIMEOUT = 30
SCROLL_PAUSE_TIME = 1.0
MAX_SCROLLS = 3
MAX_RETRIES = 5
REQUEST_DELAY = 2
DATA_DIRECTORY = "keyword_data"
CHROME_DRIVER_PATH = None

# Proxy configuration constants
PROXY_TYPE_HTTP = 'http'
PROXY_TYPE_SOCKS5 = 'socks5'
PROXY_TYPES = [PROXY_TYPE_HTTP, PROXY_TYPE_SOCKS5]

# Webdriver process management
webdriver_processes = []

# User Agent rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0"
]

# Configure Logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').addHandler(console)

link_file_lock = threading.Lock()
lock = Lock()
shutdown_event = Event()

def cleanup_webdriver_processes():
    for process in webdriver_processes:
        try:
            process.terminate()
            logging.info(f"Terminated webdriver process {process.pid}")
        except:
            pass
    try:
        if os.name == 'posix':
            os.system("pkill -f chromedriver")
            os.system("pkill -f chrome")
        elif os.name == 'nt':
            os.system("taskkill /f /im chromedriver.exe")
            os.system("taskkill /f /im chrome.exe")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
atexit.register(cleanup_webdriver_processes)

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start=9000, end=10000):
    for port in range(start, end):
        if not is_port_in_use(port):
            return port
    return None

class KeywordScraper:
    def __init__(self, base_url, keywords, file_format="csv", max_pages_per_keyword=10, driver=None, proxy=None, proxy_type=PROXY_TYPE_HTTP):
        self.driver = driver
        self.base_url = base_url
        self.keywords = keywords
        self.visited = set()
        self.file_format = file_format
        self.max_pages_per_keyword = max_pages_per_keyword
        self._driver_external = False if driver is None else True
        self.proxy = proxy
        self.proxy_type = proxy_type

        parsed_url = urlparse(base_url)
        domain_parts = parsed_url.netloc.split('.')
        self.domain = domain_parts[-2] if len(domain_parts) > 1 else parsed_url.netloc

        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        self.keyword_data = {keyword: [] for keyword in keywords}

    @backoff.on_exception(backoff.expo,
                         (WebDriverException, SessionNotCreatedException),
                         max_tries=5)
    def setup_driver(self, headless=True, port=None):
        try:
            if self.driver:
                logging.info("Using existing WebDriver")
                return self.driver

            chrome_options = Options()

            if headless:
                #chrome_options.add_argument("--headless=new")
                chrome_options.add_argument('--disable-gpu')
                chrome_options.add_argument("--window-size=1920,1080")
            
            chrome_options.add_argument("--disable-software-rasterizer")
            chrome_options.add_argument("--remote-allow-origins=*")

            if port is None:
                port = find_available_port()
                if port is None:
                    logging.error("Could not find an available port")
                    raise RuntimeError("No available ports for ChromeDriver")

            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")
            chrome_options.add_argument(f"--remote-debugging-port={port}")

            chrome_options.add_argument("--disable-infobars")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-popup-blocking")
            chrome_options.add_argument("--disable-notifications")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-setuid-sandbox")
            chrome_options.add_argument("--disable-features=TranslateUI")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--ignore-certificate-errors")
            chrome_options.add_argument("--allow-running-insecure-content")

            chrome_prefs = {
                "profile.default_content_setting_values.notifications": 2,
                "profile.default_content_settings.popups": 0,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": False,
                "plugins.always_open_pdf_externally": True
            }
            chrome_options.add_experimental_option("prefs", chrome_prefs)

            if self.proxy:
                if self.proxy_type == PROXY_TYPE_HTTP:
                    chrome_options.add_argument(f'--proxy-server=http://{self.proxy}')
                elif self.proxy_type == PROXY_TYPE_SOCKS5:
                    chrome_options.add_argument(f'--proxy-server=socks5://{self.proxy}')
                logging.info(f"Using {self.proxy_type} proxy: {self.proxy}")

            try:
                if CHROME_DRIVER_PATH:
                    self.driver = uc.Chrome(
                        options=chrome_options,
                        driver_executable_path=CHROME_DRIVER_PATH,
                        version_main=120
                    )
                else:
                    self.driver = uc.Chrome(options=chrome_options)

                logging.info(f"WebDriver set up successfully for {self.base_url} on port {port}")
                return self.driver

            except Exception as e:
                logging.error(f"Error with undetected_chromedriver: {e}")
                logging.info("Falling back to manual Chrome process...")

                chrome_process = None
                if os.name == 'posix':
                    chrome_path = "/usr/bin/google-chrome"
                    chrome_process = subprocess.Popen([
                        chrome_path,
                        f"--remote-debugging-port={port}",
                        "--disable-gpu",
                        "--headless=new",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        f'--proxy-server=http://{self.proxy}' if self.proxy and self.proxy_type == PROXY_TYPE_HTTP else '',
                        f'--proxy-server=socks5://{self.proxy}' if self.proxy and self.proxy_type == PROXY_TYPE_SOCKS5 else ''
                    ])
                    webdriver_processes.append(chrome_process)

                from selenium import webdriver
                chrome_options.add_experimental_option("debuggerAddress", f"127.0.0.1:{port}")
                self.driver = webdriver.Chrome(options=chrome_options)
                logging.info("Fallback WebDriver created successfully")
                return self.driver

        except WebDriverException as e:
            logging.error(f"Critical error setting up WebDriver: {e}")
            raise

    def close_driver(self):
        if self.driver and not self._driver_external:
            try:
                self.driver.quit()
                logging.info("WebDriver closed successfully")
            except Exception as e:
                logging.error(f"Error closing WebDriver: {e}")
                try:
                    if hasattr(self.driver, 'service') and hasattr(self.driver.service, 'process'):
                        if self.driver.service.process:
                            pid = self.driver.service.process.pid
                            if pid:
                                if os.name == 'posix':
                                    os.kill(pid, signal.SIGTERM)
                                elif os.name == 'nt':
                                    os.system(f"taskkill /F /PID {pid}")
                except:
                    pass

    @backoff.on_exception(backoff.expo,
                         (WebDriverException, TimeoutException),
                         max_tries=3)
    def scroll_page(self, timeout=BASE_TIMEOUT, max_scrolls=MAX_SCROLLS):
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            scrolls = 0
            scroll_behaviors = [
                "window.scrollTo(0, document.body.scrollHeight);",
                "window.scrollTo(0, document.body.scrollHeight * 0.7);",
                "window.scrollTo(0, document.body.scrollHeight * 0.5);"
            ]
            while scrolls < max_scrolls:
                scroll_script = random.choice(scroll_behaviors)
                try:
                    self.driver.execute_script(scroll_script)
                    time.sleep(SCROLL_PAUSE_TIME + random.uniform(0.3, 0.8))
                    new_height = self.driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        if random.random() > 0.7:
                            break
                    last_height = new_height
                    scrolls += 1
                    if random.random() > 0.7:
                        time.sleep(random.uniform(0.5, 1.0))
                except Exception as scroll_error:
                    logging.warning(f"Scroll error (continuing): {scroll_error}")
                    time.sleep(0.5)
                    break
            logging.info(f"Scrolled {scrolls} times on page")
        except Exception as e:
            logging.error(f"Error during scrolling: {e}")

    def get_page_content(self, url, timeout=BASE_TIMEOUT, retries=MAX_RETRIES):
        for attempt in range(retries):
            try:
                self.driver.get(url)
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                try:
                    self.driver.execute_script("return navigator.userAgent")
                except Exception as session_err:
                    logging.error(f"Broken WebDriver session detected: {session_err}")
                    if not self._driver_external:
                        self.close_driver()
                        self.setup_driver(headless=True)
                    else:
                        raise WebDriverException("Cannot restart externally provided driver")
                self.scroll_page()
                if random.random() > 0.7:
                    time.sleep(random.uniform(0.5, 1.2))
                html = self.driver.page_source
                if html and len(html) > 500:
                    return html
                else:
                    logging.warning(f"Retrieved suspiciously small HTML ({len(html) if html else 0} bytes)")
                    if attempt < retries - 1:
                        continue
            except TimeoutException:
                logging.warning(f"Timeout on {url}, attempt {attempt + 1}/{retries}")
                timeout = timeout * 1.5
            except WebDriverException as e:
                logging.error(f"WebDriver error on {url} (attempt {attempt + 1}/{retries}): {e}")
                if not self._driver_external and "session deleted" in str(e).lower():
                    try:
                        self.close_driver()
                        self.setup_driver(headless=True)
                    except Exception as restart_err:
                        logging.error(f"Failed to restart driver: {restart_err}")
            except Exception as e:
                logging.error(f"Error loading {url} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                wait_time = REQUEST_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logging.info(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        logging.error(f"Failed to load {url} after {retries} attempts.")
        try:
            logging.info(f"Attempting fallback with requests library for {url}")
            headers = {'User-Agent': random.choice(USER_AGENTS), 'Accept-Language': 'en-US,en;q=0.9'}
            if self.proxy and self.proxy_type == PROXY_TYPE_HTTP:
                proxies = {'http': f'http://{self.proxy}', 'https': f'http://{self.proxy}'}
                response = requests.get(url, headers=headers, timeout=timeout, proxies=proxies)
            elif self.proxy and self.proxy_type == PROXY_TYPE_SOCKS5:
                proxies = {'http': f'socks5://{self.proxy}', 'https': f'socks5://{self.proxy}'}
                response = requests.get(url, headers=headers, timeout=timeout, proxies=proxies)
            else:
                response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                logging.info(f"Fallback successful for {url}")
                return response.text
        except Exception as req_err:
            logging.error(f"Requests fallback also failed: {req_err}")
        return None

    @lru_cache(maxsize=100)
    def get_domain(self, url):
        return urlparse(url).netloc

    def extract_links(self, html_source):
        if not html_source:
            return set()
        soup = BeautifulSoup(html_source, "html.parser")
        links = set()
        base_domain = self.get_domain(self.base_url)
        base_tag = soup.find('base', href=True)
        base_url = base_tag['href'] if base_tag else self.base_url
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if not href or href.startswith("javascript:") or href == "#" or href.startswith("mailto:"):
                continue
            try:
                absolute_url = urljoin(base_url, href)
                url_parts = urlparse(absolute_url)
                clean_url = urlparse(absolute_url)._replace(fragment='').geturl()
                link_domain = self.get_domain(clean_url)
                if link_domain == base_domain:
                    links.add(clean_url)
            except Exception as e:
                logging.warning(f"Error processing link {href}: {e}")
                continue
        with link_file_lock:
            try:
                with open('extracted_links.txt', 'a') as f:
                    for link in links:
                        f.write(link + '\n')
            except Exception as e:
                logging.error(f"Error writing to links file: {e}")
        logging.info(f"Extracted {len(links)} internal links")
        return links

    def extract_data_with_keywords(self, html_source, selectors, keyword):
        if not html_source:
            return []
        try:
            soup = BeautifulSoup(html_source, 'html.parser')
            results = []
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_url = self.driver.current_url if self.driver else "unknown"
            title = soup.title.string if soup.title else "No title"
            keyword_clean = keyword.strip().lower()
            keyword_parts = keyword_clean.split()
            keyword_variations = [
                keyword_clean,
                keyword.upper(),
                keyword.capitalize(),
                ' ' + keyword_clean + ' ',
                *[part.lower() for part in keyword_parts if len(part) > 3]
            ]
            elements = []
            for selector in selectors:
                try:
                    elements.extend(soup.select(selector))
                except Exception as sel_err:
                    logging.warning(f"Error with selector {selector}: {sel_err}")

            for element in elements:
                try:
                    text = element.get_text(strip=True)
                    if not text:
                        continue
                    text_lower = text.lower()
                    found_match = False
                    matching_keyword = None
                    for variation in keyword_variations:
                        if variation in text_lower:
                            found_match = True
                            matching_keyword = variation
                            break
                    if found_match:
                        data_item = {
                            "text": text,
                            "selector": element.name,
                            "matched_keyword": matching_keyword,
                            "url": current_url,
                            "timestamp": timestamp,
                            "title": title,
                            "keyword": keyword
                        }
                        context_containers = [
                            element.find_parent(['article', 'section', 'main']),
                            element.find_parent(['div.content', 'div.main', 'div.article']),
                            element.parent
                        ]
                        for container in context_containers:
                            if not container or container == element:
                                continue
                            context = container.get_text(strip=True)
                            if len(context) > len(text) and len(context) < 2000:
                                data_item["context"] = context
                                break
                        results.append(data_item)
                except Exception as elem_err:
                    logging.warning(f"Error processing element: {elem_err}")
                    continue
            logging.info(f"Found {len(results)} items containing keyword '{keyword}'")
            return results
        except Exception as e:
            logging.error(f"Error extracting data with keyword '{keyword}': {e}")
            return []

    def save_keyword_data(self, keyword):
        safe_keyword = ''.join(c if c.isalnum() else '_' for c in keyword.lower())
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        filename = f"{safe_keyword}_{self.domain}_{timestamp}"
        if self.file_format == 'json':
            self._save_json(keyword, filename)
        elif self.file_format == 'csv':
            self._save_csv(keyword, filename)
        else:
            logging.error(f"Invalid format: {self.file_format}, defaulting to CSV")
            self._save_csv(keyword, keyword)

    def _save_json(self, keyword, filename):
        try:
            filepath = os.path.join(DATA_DIRECTORY, f"{filename}.json")
            data_to_save = {
                "keyword": keyword,
                "website": self.base_url,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "items_count": len(self.keyword_data[keyword]),
                "data": self.keyword_data[keyword]
            }
            with link_file_lock:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved {len(self.keyword_data[keyword])} items for keyword '{keyword}' to {filepath}")
        except Exception as e:
            logging.error(f"Error saving data to JSON file for keyword '{keyword}': {e}")
            try:
                emergency_path = os.path.join(DATA_DIRECTORY, f"emergency_{filename}.json")
                with open(emergency_path, "w", encoding="utf-8") as f:
                    json.dump(self.keyword_data[keyword], f, ensure_ascii=False)
                logging.info(f"Created emergency backup at {emergency_path}")
            except Exception as backup_err:
                logging.error(f"Emergency backup also failed: {backup_err}")

    def _save_csv(self, keyword, filename):
        try:
            filepath = os.path.join(DATA_DIRECTORY, f"{filename}.csv")
            if not self.keyword_data[keyword]:
                with link_file_lock:
                    with open(filepath, "w", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["keyword", "url", "title", "selector", "text", "timestamp", "context", "matched_keyword"])
                logging.info(f"Created empty CSV file for keyword '{keyword}' at {filepath}")
                return
            all_keys = set()
            for item in self.keyword_data[keyword]:
                all_keys.update(item.keys())
            fieldnames = ["keyword", "matched_keyword", "url", "title", "selector", "text", "timestamp"]
            for key in all_keys:
                if key not in fieldnames:
                    fieldnames.append(key)
            with link_file_lock:
                with open(filepath, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for item in self.keyword_data[keyword]:
                        row = {field: item.get(field, "") for field in fieldnames}
                        writer.writerow(row)
            logging.info(f"Saved {len(self.keyword_data[keyword])} items for keyword '{keyword}' to {filepath}")
        except Exception as e:
            logging.error(f"Error saving data to CSV file for keyword '{keyword}': {e}")
            try:
                emergency_path = os.path.join(DATA_DIRECTORY, f"emergency_{filename}.csv")
                with open(emergency_path, "w", encoding="utf-8", newline="") as f:
                    f.write(str(self.keyword_data[keyword]))
                logging.info(f"Created emergency text backup at {emergency_path}")
            except:
                pass

    def search_keyword(self, keyword, selectors, queue=None):
        pages_processed = 0
        links_to_visit = [self.base_url]
        local_visited = set()
        error_count = 0
        last_request_time = 0

        logging.info(f"Starting search for keyword '{keyword}' on {self.base_url}")

        try:
            while (links_to_visit and
                   pages_processed < self.max_pages_per_keyword and
                   error_count < 3 and
                   not shutdown_event.is_set()):

                current_time = time.time()
                elapsed = current_time - last_request_time
                if elapsed < REQUEST_DELAY:
                    time.sleep(REQUEST_DELAY - elapsed + random.uniform(0.1, 0.5))

                last_request_time = time.time()

                try:
                    current_url = links_to_visit.pop(0)
                except IndexError:
                    break

                with lock:
                    if current_url in self.visited:
                        continue
                    self.visited.add(current_url)

                local_visited.add(current_url)

                pages_processed += 1
                logging.info(f"Processing page {pages_processed}/{self.max_pages_per_keyword} for '{keyword}': {current_url}")

                html = self.get_page_content(current_url)
                if not html:
                    error_count += 1
                    continue

                error_count = 0

                data = self.extract_data_with_keywords(html, selectors, keyword)

                if data:
                    with lock:
                        self.keyword_data[keyword].extend(data)
                        if len(self.keyword_data[keyword]) % 10 == 0:
                            self.save_keyword_data(keyword)

                if pages_processed < self.max_pages_per_keyword:
                    new_links = self.extract_links(html)
                    for link in new_links:
                        if link in local_visited:
                            continue
                        with lock:
                            if link not in self.visited and link not in links_to_visit:
                                links_to_visit.append(link)

                time.sleep(random.uniform(REQUEST_DELAY, REQUEST_DELAY * 1.5))

            logging.info(f"Completed search for '{keyword}' on {self.base_url}. "
                         f"Processed {pages_processed} pages, found {len(self.keyword_data[keyword])} items.")
            self.save_keyword_data(keyword)
            return pages_processed

        except Exception as e:
            logging.error(f"Error searching for keyword '{keyword}': {e}")
            if self.keyword_data[keyword]:
                self.save_keyword_data(keyword)
            return pages_processed

        finally:
            if queue and links_to_visit and pages_processed < self.max_pages_per_keyword:
                for link in links_to_visit:
                    if link not in local_visited:
                        with lock:
                            if link not in self.visited:
                                queue.put((keyword, selectors, keyword))

    def run(self):
        try:
            selectors = [
                "h1", "h2", "h3", "h4", "p", "li",
                "div.content", "article", "section",
                "div.main", "div.article", "main"
            ]
            if not hasattr(self, 'driver') or not self.driver:
                self.setup_driver(headless=True)

            logging.info(f"Starting scraping process for {len(self.keywords)} keywords")
            with ThreadPoolExecutor(max_workers=min(len(self.keywords), MAX_THREADS)) as executor:
                futures = {executor.submit(self.search_keyword, keyword, selectors): keyword
                           for keyword in self.keywords}
                for future in futures:
                    keyword = futures[future]
                    try:
                        pages = future.result()
                        item_count = len(self.keyword_data[keyword])
                        logging.info(f"Completed '{keyword}': Processed {pages} pages, found {item_count} items")
                        if item_count > 0:
                            self.save_keyword_data(keyword)
                    except Exception as e:
                        logging.error(f"Error processing keyword '{keyword}': {str(e)}")
                        if keyword in self.keyword_data and self.keyword_data[keyword]:
                            self.save_keyword_data(keyword)
            self._generate_summary_report()
            logging.info("Scraping completed successfully")
            return True

        except Exception as e:
            logging.error(f"Critical error in scraper run method: {str(e)}", exc_info=True)
            self._emergency_data_save()
            return False

        finally:
            if hasattr(self, 'driver') and self.driver and not self._driver_external:
                self.close_driver()

    def _generate_summary_report(self):
        try:
            summary = {
                "scrape_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "base_url": self.base_url,
                "total_pages_visited": len(self.visited),
                "keywords": {}
            }
            for keyword in self.keywords:
                items = self.keyword_data.get(keyword, [])
                if not items:
                    summary["keywords"][keyword] = {"count": 0, "status": "No matches found"}
                    continue
                unique_urls = set(item.get("url", "") for item in items)
                summary["keywords"][keyword] = {
                    "count": len(items),
                    "unique_urls": len(unique_urls),
                    "status": "Completed",
                    "sample_matches": [item.get("text", "")[:100] + "..." for item in items[:3]]
                }
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            filename = f"summary_{self.domain}_{timestamp}.json"
            filepath = os.path.join(DATA_DIRECTORY, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logging.info(f"Summary report saved to {filepath}")
        except Exception as e:
            logging.error(f"Error generating summary report: {str(e)}")

    def _emergency_data_save(self):
        try:
            emergency_dir = os.path.join(DATA_DIRECTORY, "emergency_backup")
            os.makedirs(emergency_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filepath = os.path.join(emergency_dir, f"emergency_data_{timestamp}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.keyword_data, f, ensure_ascii=False)
            logging.info(f"Emergency data backup saved to {filepath}")
        except Exception as e:
            logging.error(f"Failed to create emergency backup: {str(e)}")


def worker(shared_data, queue, website, file_format, max_pages, selectors, driver_pool, proxy=None, proxy_type=PROXY_TYPE_HTTP):
    driver = driver_pool.get_driver()
    local_visited = set()
    try:
        while True:
            try:
                keyword, url = queue.get(timeout=5)
                if url in local_visited:
                    queue.task_done()
                    continue
                local_visited.add(url)
                html = None
                try:
                    driver.get(url)
                    WebDriverWait(driver, BASE_TIMEOUT).until(
                        EC.presence_of_element_located((By.TAG_NAME, 'body'))
                    )
                    last_height = driver.execute_script("return document.body.scrollHeight")
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(SCROLL_PAUSE_TIME)
                    html = driver.page_source
                except Exception as e:
                    logging.error(f"Worker error loading page {url}: {e}")
                if html:
                    soup = BeautifulSoup(html, 'html.parser')
                    results = []
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    title = soup.title.string if soup.title else "No title"
                    for selector in selectors:
                        for element in soup.select(selector):
                            text = element.get_text(strip=True)
                            if keyword.lower() in text.lower():
                                data_item = {
                                    "text": text,
                                    "selector": selector,
                                    "url": url,
                                    "timestamp": timestamp,
                                    "title": title,
                                    "keyword": keyword
                                }
                                if element.parent:
                                    context = element.parent.get_text(strip=True)
                                    if len(context) > len(text):
                                        data_item["context"] = context
                                results.append(data_item)
                    if results:
                        with lock:
                            if keyword not in shared_data:
                                shared_data[keyword] = []
                            shared_data[keyword].extend(results)
                    links = set()
                    for a_tag in soup.find_all("a", href=True):
                        href = a_tag["href"]
                        absolute_url = urljoin(website, href)
                        if urlparse(absolute_url).netloc == urlparse(website).netloc:
                            links.add(absolute_url)
                    for link in links:
                        if link not in local_visited:
                            with lock:
                                queue.put((keyword, link))
                queue.task_done()
                time.sleep(REQUEST_DELAY)
            except Empty:
                break
            except Exception as e:
                logging.error(f"Worker error: {e}")
                queue.task_done()
    finally:
        driver_pool.return_driver(driver)

class DriverPool:
    """A pool of WebDriver instances for concurrent scraping."""

    def __init__(self, pool_size=MAX_THREADS, headless=True, proxy=None, proxy_type=PROXY_TYPE_HTTP):
        """Initialize the driver pool."""
        self.pool_size = pool_size
        self.headless = headless
        self.drivers = []
        self.available_drivers = Queue()
        self.lock = Lock()
        self.proxy = proxy
        self.proxy_type = proxy_type
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

        if self.headless:
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument("--window-size=1920,1080")

        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")

        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")

        if self.proxy:
            if self.proxy_type == PROXY_TYPE_HTTP:
                chrome_options.add_argument(f'--proxy-server=http://{self.proxy}')
            elif self.proxy_type == PROXY_TYPE_SOCKS5:
                chrome_options.add_argument(f'--proxy-server=socks5://{self.proxy}')
            logging.info(f"Driver in pool using {self.proxy_type} proxy: {self.proxy}")


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


def multi_threaded_keyword_search(website, keywords, selectors, file_format, max_pages, max_threads=MAX_THREADS, proxy=None, proxy_type=PROXY_TYPE_HTTP):
    queue = Queue()
    shared_data = {}
    driver_pool = DriverPool(pool_size=max_threads, headless=True, proxy=proxy, proxy_type=proxy_type)
    try:
        for keyword in keywords:
            queue.put((keyword, website))
            shared_data[keyword] = []
        threads = []
        for _ in range(max_threads):
            thread = Thread(target=worker, args=(shared_data, queue, website, file_format, max_pages, selectors, driver_pool, proxy, proxy_type))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        queue.join()
        for thread in threads:
            thread.join(timeout=1)
        driver = driver_pool.get_driver()
        scraper = KeywordScraper(website, keywords, file_format, max_pages, driver=driver, proxy=proxy, proxy_type=proxy_type)
        scraper._driver_external = True
        for keyword in keywords:
            scraper.keyword_data[keyword] = shared_data[keyword]
            scraper.save_keyword_data(keyword)
        driver_pool.return_driver(driver)
    finally:
        driver_pool.close_all()


def main():
    """Main function to run the scraper with enhanced argument parsing and proxy support."""
    global MAX_THREADS, BASE_TIMEOUT, SCROLL_PAUSE_TIME, MAX_SCROLLS, MAX_RETRIES, REQUEST_DELAY # Global declaration moved to the top of main()

    parser = argparse.ArgumentParser(description='Keyword-based web scraper with proxy support and enhanced options.')
    parser.add_argument('--website', type=str, required=True, help='Website URL to scrape.')
    parser.add_argument('--keywords', type=str, nargs='+', required=True, help='Keywords to search for.')
    parser.add_argument('--threads', type=int, default=MAX_THREADS, help=f'Number of threads to use (default: {MAX_THREADS}).')
    parser.add_argument('--format', type=str, choices=['json', 'csv'], default='csv', help='Output file format (json or csv).')
    parser.add_argument('--max-pages', type=int, default=10, help='Maximum pages to scrape per keyword.')
    parser.add_argument('--headless', action='store_true', default=True, help='Run in headless mode (no browser window).')
    parser.add_argument('--no-headless', dest='headless', action='store_false', help='Run in non-headless mode (show browser window).')
    parser.add_argument('--selectors', type=str, nargs='+', default=["h1", "h2", "h3", "p", "li", "div.content", "article", "section"], help='CSS selectors to search for keywords.')
    parser.add_argument('--request-delay', type=float, default=REQUEST_DELAY, help=f'Delay between requests in seconds (default: {REQUEST_DELAY}s).')
    parser.add_argument('--base-timeout', type=int, default=BASE_TIMEOUT, help=f'Base timeout for page load in seconds (default: {BASE_TIMEOUT}s).')
    parser.add_argument('--max-retries', type=int, default=MAX_RETRIES, help=f'Maximum retries for page load failures (default: {MAX_RETRIES}).')
    parser.add_argument('--scroll-pause-time', type=float, default=SCROLL_PAUSE_TIME, help=f'Pause time after scrolling in seconds (default: {SCROLL_PAUSE_TIME}s).')
    parser.add_argument('--max-scrolls', type=int, default=MAX_SCROLLS, help=f'Maximum scrolls per page (default: {MAX_SCROLLS}).')

    proxy_group = parser.add_argument_group('Proxy Settings')
    proxy_group.add_argument('--proxy', type=str, help='Proxy address (e.g., ip:port).')
    proxy_group.add_argument('--proxy-type', type=str, choices=PROXY_TYPES, default=PROXY_TYPE_HTTP, help=f'Proxy type: {PROXY_TYPES} (default: {PROXY_TYPE_HTTP}).')

    args = parser.parse_args()

    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=log_format)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger('').addHandler(console)

    logging.info(f"Starting keyword scraper for website: {args.website}")
    logging.info(f"Keywords to search: {args.keywords}")
    logging.info(f"Threads: {args.threads}, Format: {args.format}, Max Pages: {args.max_pages}, Headless: {args.headless}")
    logging.info(f"Request Delay: {args.request_delay}, Base Timeout: {args.base_timeout}, Max Retries: {args.max_retries}, Scroll Pause Time: {args.scroll_pause_time}, Max Scrolls: {args.max_scrolls}")
    if args.proxy:
        logging.info(f"Proxy enabled: {args.proxy}, Type: {args.proxy_type}")
    else:
        logging.info("Proxy disabled.")


    MAX_THREADS = args.threads
    BASE_TIMEOUT = args.base_timeout
    SCROLL_PAUSE_TIME = args.scroll_pause_time
    MAX_SCROLLS = args.max_scrolls
    MAX_RETRIES = args.max_retries
    REQUEST_DELAY = args.request_delay


    try:
        multi_threaded_keyword_search(
            args.website,
            args.keywords,
            args.selectors,
            args.format,
            args.max_pages,
            args.threads,
            args.proxy,
            args.proxy_type
        )
        logging.info("Scraping complete!")
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        return 1
    return 0


if __name__ == '__main__':
    exit(main())