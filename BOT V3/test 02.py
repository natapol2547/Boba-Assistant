import re

def remove_image_links(urls):
    """
    This function takes in a list of URLs and returns a new list with any links to images removed.
    """
    image_types = ["jpg", "jpeg", "png", "gif", "bmp"]
    new_urls = []
    for url in urls:
        # Use regular expression to extract file extension
        match = re.search(r'\.([a-zA-Z0-9]+)$', url)
        if match and match.group(1).lower() in image_types:
            # This is a link to an image file, so we'll skip it
            continue
        else:
            # This is not an image link, so we'll keep it
            new_urls.append(url)
    return new_urls

my_urls = [
    "https://example.com",
    "https://example.com/image.jpg",
    "https://example.com/document.pdf",
    "https://example.com/image.png",
    "https://example.com/other.html",
]

new_urls = remove_image_links(my_urls)
print(new_urls)
