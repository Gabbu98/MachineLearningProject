import time

from bs4 import BeautifulSoup
from numpy.core.defchararray import isdigit
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.options import Options
import re

URL = "https://avpay.aero/aircraft-for-sale/single-engine-piston-airplane-for-sale/"
# Configure Chrome options for headless mode
chrome_options = Options()
chrome_options.add_argument("--headless")
dollar_pattern = re.compile("\$[0-9]+,[0-9]+")
euro_pattern = re.compile("€[0-9]+,[0-9]+")
pound_pattern = re.compile("£[0-9]+,[0-9]+")
digit_pattern = re.compile("[0-9]+,[0-9]+")


def load_page():
    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 10)
    driver.get(URL)
    next_page = 2
    page_limit = 48
    wait.until(EC.url_to_be(URL))
    with open('./data/aircraft-data.txt', 'w', encoding='utf-8') as f:
        for i in range(next_page, page_limit + 1):
            time.sleep(10)
            try:
                print("-----------------Scraping Page " + str(i - 1) + "-----------------")
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

                    list = []

                    # read from td[1]
                    for test2 in tds[1].find("div", {"data-wcpt-taxonomy": "aircraft_filter"}).find_all('div'):
                        list.append(test2.text)

                    type = list[-1]
                    year = tds[1].find("div", {"data-wcpt-taxonomy": "year_"}).find("div").text
                    seats = tds[1].find("div", {"data-wcpt-taxonomy": "seats"}).find("div").text
                    total_time = tds[1].find("div", {"data-wcpt-taxonomy": "total_time"}).find("div").text
                    price = tds[2].find("div", {"data-wcpt-taxonomy": "display_price"}).find("div").text

                    f.write(type + "|" + year + "|" + seats + "|" + total_time + "|" + price + "\n")

                # Wait for the element to be clickable (if needed)
                element = WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.XPATH, xpath_expression))
                )

                # Click on the element
                element.click()

            except Exception as e:
                print(f"Error: {e}")
                f.close()
                driver.quit()
                exit()

    # Close the browser after clicking
    driver.quit()
    f.close()


def cleanup_data():
    with open("./data/aircraft-data.txt", encoding='utf-8') as f:

        raw_data = f.read().splitlines()
        updated_data = []
        for line in raw_data:
            entries = line.split("|")

            line_to_insert = []
            for entry in entries:

                phrase = entry.split()

                if (dollar_pattern.search(entry) or euro_pattern.search(entry) or pound_pattern.search(entry)) and len(
                        phrase) > 1:

                    for word in phrase:

                        if dollar_pattern.match(word):
                            resultant_string = word[1:]
                            line_to_insert.append(resultant_string)
                            break

                        elif euro_pattern.match(word):
                            resultant_string = word[1:]
                            line_to_insert.append(resultant_string)
                            break

                        elif pound_pattern.match(word):
                            resultant_string = word[1:]
                            line_to_insert.append(resultant_string)
                            break

                        else:
                            line_to_insert.append(entry)
                            break


                else:
                    if dollar_pattern.match(entry):
                        resultant_string = entry[1:]
                        line_to_insert.append(resultant_string)

                    elif euro_pattern.match(entry):
                        resultant_string = entry[1:]
                        line_to_insert.append(resultant_string)

                    elif pound_pattern.match(entry):
                        resultant_string = entry[1:]
                        line_to_insert.append(resultant_string)

                    else:
                        line_to_insert.append(entry)

            updated_data.append(line_to_insert)

        for entry in updated_data:
            is_training = False
            line = ""
            for word in entry:
                if digit_pattern.match(word) and word == entry[-1]:
                    is_training = True

                line = line + word + "|"

            line = line + "\n"

            if is_training:
                with open('./data/aircraft-data-training.txt', 'a', encoding='utf-8') as training_file:
                    training_file.write(line)
                    training_file.close()

            else:
                with open('./data/aircraft-data-test.txt', 'a', encoding='utf-8') as test_file:
                    test_file.write(line)
                    test_file.close()
    print()
    f.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # load_page()
    cleanup_data()
