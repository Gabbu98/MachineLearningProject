from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait

URL = "https://avpay.aero/aircraft-for-sale/single-engine-piston-airplane-for-sale/"
def scrape():
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 10)
    driver.get(URL)
    get_url = driver.current_url
    wait.until(EC.url_to_be(URL))
    if get_url == URL:
        page_source = driver.page_source
    soup = BeautifulSoup(page_source, features="html.parser")

    try:
        # Construct the XPath expression to find the element by class name and text
        class_name = 'page-numbers'  # Replace with the actual class name
        element_text = '2'  # Replace with the actual text of the element
        xpath_expression = f'//a[contains(@class, "{class_name}") and contains(text(), "{element_text}")]'

        # Wait for the element to be clickable (if needed)
        element = WebDriverWait(driver, 10).until(
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