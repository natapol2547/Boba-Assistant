from deep_translator import GoogleTranslator

from bs4 import BeautifulSoup

from tqdm import tqdm
# translator = GoogleTranslator(target='en')
mylist = ['Hello world', 'How are you doing today']

translation = GoogleTranslator(source='en', target='fr').translate_batch(tqdm(mylist, desc="Translating text"))
print(translation)

from langchain.document_loaders import UnstructuredURLLoader
urls = [
    "https://www.mytcas.com/universities"
]
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
print(data)


import requests
from bs4 import BeautifulSoup
import asyncio

async def get_links_from_url(url, iterations=1):
    url_list = [url]

    for i in range(iterations):
        new_urls = []
        for url in url_list:
            grab = requests.get(url, verify=False)
            soup = BeautifulSoup(grab.text, 'html.parser')

            for link in soup.find_all("a"):
                data = link.get('href')
                if data and (data.startswith('http://') or data.startswith('https://')):
                    new_urls.append(data)
                elif data and data.startswith('/'):
                    new_urls.append(url + data)

        url_list += new_urls

    return url_list

async def run_concurrent():
    urls = ['https://www.google.com', 'https://www.youtube.com', 'https://www.facebook.com']
    tasks = []
    for url in urls:
        task = asyncio.ensure_future(get_links_from_url(url, iterations=1))
        tasks.append(task)

    responses = await asyncio.gather(*tasks)
    return responses

results = asyncio.run(run_concurrent())
print(results)
