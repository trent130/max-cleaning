from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, SessionNotCreatedException
import time
import random
import logging
import undetected_chromedriver as uc
import os
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import json
import csv

def setup_logging():
    logging.basicConfig(
        filename='keyword_scraper.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(console)

class KeywordScraper:
    def __init__(self, base_url, keywords, max_pages=10, headless=True):
        self.base_url = base_url
        self.keywords = keywords
        self.max_pages = max_pages
        self.headless = headless
        self.visited = set()
        self.data = {keyword: [] for keyword in keywords}
        self.driver = self.setup_driver()

    def setup_driver(self):
        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-infobars")
            driver = uc.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)
            driver.implicitly_wait(10)
            logging.info("WebDriver initialized successfully")
            return driver
        except WebDriverException as e:
            logging.error(f"WebDriver initialization failed: {e}")
            return None

    def get_page_content(self, url):
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(random.uniform(1, 2))
            return self.driver.page_source
        except TimeoutException:
            logging.warning(f"Timeout on {url}")
        except WebDriverException as e:
            logging.error(f"WebDriver error on {url}: {e}")
        return None

    def extract_links(self, html):
        soup = BeautifulSoup(html, "html.parser")
        links = {urljoin(self.base_url, a['href']) for a in soup.find_all("a", href=True)}
        return links

    def extract_data(self, html, keyword):
        soup = BeautifulSoup(html, "html.parser")
        results = [elem.get_text(strip=True) for elem in soup.find_all(string=lambda text: keyword.lower() in text.lower())]
        return results

    def search_keyword(self, keyword):
        to_visit = [self.base_url]
        pages_scraped = 0

        while to_visit and pages_scraped < self.max_pages:
            url = to_visit.pop(0)
            if url in self.visited:
                continue
            self.visited.add(url)
            pages_scraped += 1
            logging.info(f"Scraping {url} for '{keyword}'")
            html = self.get_page_content(url)
            if not html:
                continue
            self.data[keyword].extend(self.extract_data(html, keyword))
            to_visit.extend(self.extract_links(html))

    def run(self):
        with ThreadPoolExecutor(max_workers=min(len(self.keywords), 5)) as executor:
            executor.map(self.search_keyword, self.keywords)
        logging.info("Scraping completed")
        self.save_data()

    def save_data(self):
        os.makedirs("scraped_data", exist_ok=True)
        for keyword, items in self.data.items():
            file_path = os.path.join("scraped_data", f"{keyword}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2)
            logging.info(f"Saved data for '{keyword}' to {file_path}")

if __name__ == '__main__':
    setup_logging()
    scraper = KeywordScraper("https://en.wikipedia.org", ["war", "history"], max_pages=1000, headless=True)
    scraper.run()
