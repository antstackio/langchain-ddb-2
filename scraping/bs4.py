import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

visited_urls = set()

def scrape_and_save(url, base_url, output_dir):
    if url in visited_urls:
        return
    visited_urls.add(url)

    response = requests.get(url)
    if response.status_code != 200:
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    page_content = soup.get_text()

    # Create a unique filename based on the URL
    parsed_url = urlparse(url)
    filename = os.path.join(output_dir, f"{parsed_url.netloc}{parsed_url.path}".replace('/', '_') + '.txt')

    # Save the page content to a .txt file
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(page_content)

    # Find and process all links on the current page
    for link in soup.find_all('a', href=True):
        href = link['href']
        full_url = urljoin(base_url, href)
        if base_url in full_url:
            scrape_and_save(full_url, base_url, output_dir)

def main(base_url, output_dir='scraped_pages'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scrape_and_save(base_url, base_url, output_dir)

if __name__ == "__main__":
    website_url = "https://www.example.com"
    main(website_url)
