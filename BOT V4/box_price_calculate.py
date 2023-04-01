from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from deep_translator import GoogleTranslator
import time

options = Options()
# options.add_argument('--headless')
options.add_argument("--proxy-server='direct://'")
options.add_argument("--proxy-bypass-list=*")
options.add_argument("window-sized=1024,1324")
options.add_argument('blink-settings=imagesEnabled=false')
options.add_argument('--disable-javascript')
options.add_argument('--log-level=3') # Suppress console warnings
driver = webdriver.Chrome(options=options) # or other browser driver

def translate_string(string, source_language, target_language):
    return GoogleTranslator(source=source_language, target=target_language).translate(string)

def get_box_price(string) :
    

    # Set up the driver in headless mode
    
    print(f"Price calculation: {string}")
    try:
        width, length, height, amount, type = string.replace(' ', '').split(",")
        width = width.replace(' ', '')
        length = length.replace(' ', '')
        height = height.replace(' ', '')
        amount = amount.replace(' ', '')
    except:
        return "Wrong format of Action Input used. Use `width, length, height, amount` format."
    try:
        # Navigate to the website
        driver.get('https://box-estimate.vercel.app/')
        # time.sleep(1)
        # Find the input field and change its value to "999"
        input_field = driver.find_element(By.XPATH, '//input[@value="6"]')
        input_field.send_keys(Keys.CONTROL + 'a') # select all text in the input field
        input_field.send_keys("999") # enter new value

        # Find the input field and change its value to "999"
        input_field = driver.find_element(By.XPATH, '//input[@value="2"]')
        input_field.send_keys(Keys.CONTROL + 'a') # select all text in the input field
        input_field.send_keys("998") # enter new value

        # Find the input field and change its value to "999"
        input_field = driver.find_element(By.XPATH, '//input[@value="9.5"]')
        input_field.send_keys(Keys.CONTROL + 'a') # select all text in the input field
        input_field.send_keys("997") # enter new value

        # Find the input field and change its value to "999"
        input_field = driver.find_element(By.XPATH, '//input[@value="100"]')
        input_field.send_keys(Keys.CONTROL + 'a') # select all text in the input field
        input_field.send_keys("996") # enter new value
        
        input_field = driver.find_element(By.XPATH, '//input[@value="999"]')
        input_field.send_keys(Keys.CONTROL + 'a') # select all text in the input field
        input_field.send_keys(width) # enter new value

        # Find the input field and change its value to "999"
        input_field = driver.find_element(By.XPATH, '//input[@value="998"]')
        input_field.send_keys(Keys.CONTROL + 'a') # select all text in the input field
        input_field.send_keys(length) # enter new value

        # Find the input field and change its value to "999"
        input_field = driver.find_element(By.XPATH, '//input[@value="997"]')
        input_field.send_keys(Keys.CONTROL + 'a') # select all text in the input field
        input_field.send_keys(height) # enter new value

        # Find the input field and change its value to "999"
        input_field = driver.find_element(By.XPATH, '//input[@value="996"]')
        input_field.send_keys(Keys.CONTROL + 'a') # select all text in the input field
        input_field.send_keys(amount) # enter new value

        # Find the button and click it
        calculate_span = driver.find_element(By.XPATH, '//span[text()="Calculate"]')
        calculate_button = calculate_span.find_element(By.XPATH, './..')
        calculate_button.click()

        # Find the root div that contains the text "ขนาดกระดาษ :"
        root_div = driver.find_element(By.XPATH, '//span[text()="ขนาดกระดาษ :"]/../../..')

        # Print all the text in the root div
        div_text = root_div.text.replace('\n', ' ').replace('Calculate ', '').replace('ขนาดกางออก ', '\nขนาดกางออก').replace('พื้นที่ทั้งหมด ', '\nพื้นที่ทั้งหมด').replace('ราคารวม ', '\nราคารวม')
        
        div_text = translate_string(div_text, 'th', 'en')
        # driver.quit()
        # print(div_text)
        if "ไม่เกินขนาด A3" in div_text:
            return "Box size too big. Size must not be more than A4. Please use a smaller dimensions."
        return div_text
    except:
        return "Contacting website failed."


start_time = time.time()
print(get_box_price("7,5,5,1000,box"))
print(get_box_price("7,5,5,1000,box"))
end_time = time.time()
total_time = end_time - start_time
print(f"Program took {total_time:.2f} seconds to run.")
