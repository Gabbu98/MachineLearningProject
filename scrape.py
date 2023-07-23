import time

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.options import Options

URL = "https://avpay.aero/aircraft-for-sale/single-engine-piston-airplane-for-sale/"
# Configure Chrome options for headless mode
chrome_options = Options()
chrome_options.add_argument("--headless")

def scrape():
    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 10)
    driver.get(URL)
    next_page = 2
    page_limit = 48
    wait.until(EC.url_to_be(URL))

    for i in range(next_page,page_limit+1):
        time.sleep(10)
        try:
            # Construct the XPath expression to find the element by class name and text
            class_name = 'page-numbers'  # Replace with the actual class name
            element_text = str(i)  # Replace with the actual text of the element
            xpath_expression = f'//a[contains(@class, "{class_name}") and contains(text(), "{element_text}")]'

            soup = BeautifulSoup(driver.page_source, features="html.parser")

            # Find the tbody tag
            tbody = soup.find('tbody')

            # Extract all tr tags from the tbody
            trs = tbody.find_all('tr')

            # Process each tr tag and its content
            for tr in trs:
                # Example: Get the text from each td within the tr
                tds = tr.find_all('td')
                for td in tds:
                    print(td.text)

            # Wait for the element to be clickable (if needed)
            element = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, xpath_expression))
            )

            # Click on the element
            element.click()

        except Exception as e:
            print(f"Error: {e}")

    # Close the browser after clicking
    driver.quit()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    scrape()