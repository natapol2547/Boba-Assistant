import re

def find_urls(text):
    # Regular expression pattern to match URLs
    pattern = r'(https?://[^\s]+)'
    
    # Use re.findall to find all matches of the pattern in the text
    urls = re.findall(pattern, text)
    
    urls = [url.rstrip('.') for url in urls]
# print(urls)

    # Return the list of URLs
    return urls



text = "You can see all the images here: https://www.thaiprintshop.com/collection/box/%E0%B8%AB%E0%B8%A5%E0%B8%B9%E0%B9%88-box/"
urls = find_urls(text)
print(urls)
