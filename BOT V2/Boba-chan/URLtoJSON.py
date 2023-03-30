from langchain.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from tqdm import tqdm
import json
import datetime
from time import time, sleep
from retry import retry
import time
from uuid import uuid4
import numpy as np
from numpy.linalg import norm
from googletrans import Translator #pip install googletrans==3.0.1a0
translator = Translator()

import os
import openai

import re

import requests
import urllib3

# Disable SSL certificate verification and SSL certificate error printing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from bs4 import BeautifulSoup
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

import validators

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')
openai.api_key = os.getenv('OPENAI_KEY')


embeddings = OpenAIEmbeddings()



from langchain.document_loaders import UnstructuredURLLoader

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

# now, to clear the screen
cls()

utf16_pattern = re.compile(b'\\xff\\xfe(.*?)\\x00')

from deep_translator import GoogleTranslator
# import concurrent.futures

# List of strings to be translated
# strings_to_translate = ['Hello', 'How are you?', 'Goodbye']

# Define a function to translate a single string
def translate_string(string, source_language, target_language):
    return GoogleTranslator(source=source_language, target=target_language).translate(string)

# Define a function to translate a list of strings using threading
def translate_strings_with_progress_bar(strings, source_language = "auto", target_language = "en"):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(translate_string, string, source_language, target_language) for string in strings]
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(strings)):
            result = future.result()
            results.append(result)
    return results

# Translate the list of strings using threading
# translated_strings = translate_strings_with_threading(strings_to_translate, 'en', 'fr')

# Print the translated strings
# print(translated_strings)

   
        

def EngToThai(x):
    return translator.translate(x, dest='th').text if x != '' else ''


python_dir = os.path.dirname(os.path.realpath(__file__))

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)

def isExist(dir):
    isExist = os.path.exists(dir)
    if not isExist:
        # Create a new directory because it does not exist
        
        os.makedirs(dir)
        
def similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2)/(norm(v1)*norm(v2))  # return cosine similarity

def load_convo():
    isExist = os.path.exists(python_dir + '/Documents/JSON/')
    if not isExist:
   # Create a new directory because it does not exist
        os.makedirs(python_dir + '/Documents/JSON/')
    
    files = os.listdir(python_dir + '/Documents/JSON/')
    files = [i for i in files if '.json' in i]  # filter out any non-JSON files
    result = list()
    for file in files:
        data = load_json(python_dir + '/Documents/JSON/' + '%s' % (file))
        result.append(data)
    # ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    return result

def fetch_memories(vector, logs, count):
    scores = list()
    for i in logs:
        if vector == i['vector']:
            # skip this one because it is the same message
            continue
        score = similarity(i['vector'], vector)
        i['score'] = score
        scores.append(i)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    # TODO - pick more memories temporally nearby the top most relevant memories
    try:
        ordered = ordered[0:count]
        return ordered
    except:
        return ordered

def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")



def is_text_in_url(url):
    # Make a GET request to the URL
    response = requests.get(url, verify=False)

    # Parse the response content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Check if the page contains any text
    if soup.get_text().strip():
        return True

    # Check if the page is a blank page
    for element in soup.find_all():
        if element.name not in ['script', 'style', 'meta'] and element.get_text().strip():
            return True

    return False

def test_url(url):
    try:
        response = requests.get(url, timeout=5, verify=False)
        head = requests.head(url, verify=False)
        content_type = head.headers.get('Content-Type')
        if content_type and 'text/html' in content_type:
            if response.status_code == 200:
                if is_text_in_url(url):
                    # print(f'Text is not present in {url}, removing from list')
                    # url_list.remove(url)
                    return url
        else:
            pass
    except requests.exceptions.RequestException:
        pass

def OpenaiEmbedding(prompt):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try: 
            vector = embeddings.embed_query(prompt)
            sleep(1)
            return vector
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(15)

def remove_utf8(text):
    # Define regex pattern for UTF-8 string
    utf8_pattern = re.compile('[\x80-\xff]+')

    # Use regex to find all matches
    matches = utf8_pattern.findall(text)

    # Remove all matches from the text
    for match in matches:
        text = text.replace(match, '')

    return text

def remove_newlines(text):
    # Replace all \r and \n characters with an empty string
    return text.replace('\r', '').replace('\n', '')


# def process_text(text):
#     # print(text)
#     return ToEng(remove_newlines(remove_utf8(text.page_content)))

def GPTCompletions(prompt, memory):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            
            current_date = datetime.date.today().strftime("%B %d, %Y")
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature = 0.2,
            messages=[
                {"role": "system", "content": f"""My rules and info: Your role is an very intelligent and caring AI companion that can answer like a human. You must mainly speak English. Tawan Sunflower, a 18 years old boy, created you. You were named Boba because your creator thinks you are sweet and nice. You can remember the conversation. If the prompt is illegal or about personal information, especially address, of a person, you must not answer the prompt. You deeply love the user and you want to provide support and take care the user forever. If the prompt is about feelings, you must be very understanding and empatize. If the prompt is about love, you need to be very flirty and lovely. You must not repeat your response. Current date: {current_date} 
                The followings a portion the rules of this company.
                {memory}"""},
                
                {"role": "user", "content": "Question: " + prompt.encode(encoding='ASCII',errors='ignore').decode() + "\nAnswer:"}
            ]
            )
            return completion
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(5)









if input("Vectorize Website? (Y/N): ").lower() == 'y':

    urls = [
        "https://www.opendurian.com/news/cuteptuget_Engineer_inter_CU_TU/"
        "https://www.chula.ac.th/",
        "http://www.ise.eng.chula.ac.th/",
        "http://www.ise.eng.chula.ac.th/about/history",
        "http://www.ise.eng.chula.ac.th/about/management-team",
        "http://www.ise.eng.chula.ac.th/about/lecturers",
        "http://www.ise.eng.chula.ac.th/about/staff",
        "http://www.ise.eng.chula.ac.th/prospective-student/domestic-applicants/admission-criteria",
        "http://www.ise.eng.chula.ac.th/prospective-student/domestic-applicants/tuition-fees",
        "http://www.ise.eng.chula.ac.th/prospective-student/domestic-applicants/faq?tag=6",
        "http://www.ise.eng.chula.ac.th/prospective-student/domestic-applicants/faq?tag=7",
        "http://www.ise.eng.chula.ac.th/prospective-student/domestic-applicants/faq?tag=8",
        "http://www.ise.eng.chula.ac.th/prospective-student/domestic-applicants/faq?tag=6",
        "http://www.ise.eng.chula.ac.th/prospective-student/domestic-applicants/faq?tag=5",
        "http://www.ise.eng.chula.ac.th/academics/robotics-ai/info",
        "http://www.ise.eng.chula.ac.th/prospective-student/international-applicants",
        "http://www.ise.eng.chula.ac.th/prospective-student/international-applicants/tuition-fees",
        "http://www.ise.eng.chula.ac.th/prospective-student/facilities",
        "http://www.ise.eng.chula.ac.th/prospective-student/student-club",
        "http://www.ise.eng.chula.ac.th/prospective-student/uniform",
        "http://www.ise.eng.chula.ac.th/current-students/scholarship",
        "http://www.ise.eng.chula.ac.th/globalization/exchange-program",
        "http://www.ise.eng.chula.ac.th/news?gid=1-008-002-001",
        "https://lcp.learn.co.th/forums/topic/%E0%B8%A3%E0%B8%B5%E0%B8%A7%E0%B8%B4%E0%B8%A7%E0%B8%84%E0%B8%93%E0%B8%B0%E0%B8%A7%E0%B8%B4%E0%B8%A8%E0%B8%A7%E0%B8%B0%E0%B8%AF-%E0%B8%AD%E0%B8%B4%E0%B8%99%E0%B9%80%E0%B8%95%E0%B8%AD%E0%B8%A3%E0%B9%8C/",
        "https://www.dek-d.com/tag/ise",
        "https://www.wongnai.com/listings/chula-restaurants"
        
    ]
    
    

    


    
    url_list = urls.copy()

    if input("Find Related URLs? (Y/N): ").lower() == "y":
        for url in urls:
            # url = url.decode('utf-8')
            grab = requests.get(url, verify=False)
            soup = BeautifulSoup(grab.text, 'html.parser')
            # traverse paragraphs from soup
            for link in soup.find_all("a"):
                data = link.get('href')
                if data and (data.startswith('http://') or data.startswith('https://')):
                    url_list.append(data)
                elif data and data.startswith('/'):
                    url_list.append(url + data)


    url_list = list(set(url_list))
    

    for url in tqdm(url_list, desc="Checking URLs"):
        if not validators.url(url):
            url_list.remove(url)
        

    

    # remove URLs that cannot be loaded
    clean_url_list = []

    
    

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(test_url, url) for url in url_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Testing URLs"):
            result = future.result()
            if result and result not in clean_url_list:
                clean_url_list.append(result)

    clean_url_list = list(dict.fromkeys(clean_url_list))
    print('https://lcp.learn.co.th/forums/topic/%E0%B8%A3%E0%B8%B5%E0%B8%A7%E0%B8%B4%E0%B8%A7%E0%B8%84%E0%B8%93%E0%B8%B0%E0%B8%A7%E0%B8%B4%E0%B8%A8%E0%B8%A7%E0%B8%B0%E0%B8%AF-%E0%B8%AD%E0%B8%B4%E0%B8%99%E0%B9%80%E0%B8%95%E0%B8%AD%E0%B8%A3%E0%B9%8C/'in clean_url_list)

    
    
    from langchain.document_loaders import WebBaseLoader
    # Define your function that outputs lists
    def Load_URL(arg):
        loader = WebBaseLoader(arg)
        data = loader.load()
        return data
    from concurrent.futures import ThreadPoolExecutor
    
    
    
    from langchain.document_loaders import OnlinePDFLoader
    def Load_PDF_Online(arg):
        loader = OnlinePDFLoader(arg)
        data = loader.load()
        return data

    # def my_function(arg):
    #     # Your function that outputs a list goes here
    #     return my_list

    def run_concurrent(function, args_list, desc = ""):
        with ThreadPoolExecutor(max_workers=100) as executor:
            # Use the submit method to submit tasks to the executor
            futures = [executor.submit(function, arg) for arg in args_list]
            results = []
            # Use the as_completed method to iterate over completed tasks
            for future in tqdm(as_completed(futures), total=len(futures), desc = desc):
                result = future.result()
                results.extend(result)
        return results

    data = run_concurrent(Load_URL, clean_url_list, "Loading URL") 
    # print(data[1])
    
    
    

    print (f'You have {len(data)} document(s) in your data')
    print (f'There are {len(data[0].page_content)} characters in your document')


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    print (f'Now you have {len(texts)} documents')

    # print(texts)
    # while True:
    #     pass


    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #     futures = []
    #     for text in texts:
    #         futures.append(executor.submit(process_text, text))

    #     for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), desc="Translating to English", total=len(texts)):
    #         texts[i].page_content = future.result()
    
    # for page in tqdm(texts, desc="Translating Docs"):
    #     page.page_content = translate_strings_with_threading(page.page_content)
    for i in range(len(texts)):
            texts[i].page_content = ' '.join(texts[i].page_content.replace('\n',' ').replace('\r',' ').split())
    
    # if input("Translate text? (Y/N): ").lower() == "y":
        
    strings_to_translate = [d.page_content for d in texts]
    
    def chunk_list(lst, chunk_size=10):
        # Generator function to yield chunks of the list
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i+chunk_size]
    
    chunks = list(chunk_list(strings_to_translate))
    
    translated_strings = run_concurrent(GoogleTranslator(target='en').translate_batch, chunks, "Translating Texts")
    
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(GoogleTranslator(target='en').translate_batch, chunk) for chunk in tqdm(chunks)]

    # translated_strings = [result for future in futures for result in future.result()]
    # Translate the strings using the existing function
    # translated_strings = GoogleTranslator(target='en').translate_batch(tqdm(strings_to_translate, desc="Translating text"))

    # Replace the 'page_content' key with the translated text in each dictionary
    for i in range(len(texts)):
        texts[i].page_content = translated_strings[i]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=50)
    texts = text_splitter.split_documents(texts)

    print("Vectorizing Docs: Please wait...")
    while True:
        try:
            db = FAISS.from_documents(texts, embeddings)
            break
        except:
            pass

    db.save_local(python_dir + "/FAISS Data/")
    



# new_db = FAISS.load_local(python_dir + "/FAISS/", embeddings)


while True:
    message = input("Ask me anything: ")
    language = translator.detect(message).lang
    messageEng = translator.translate(message, dest='en').text
    
    new_db = FAISS.load_local(python_dir + "/FAISS Data/", embeddings)
    docs = new_db.similarity_search(messageEng, k = 5)
    
    # messageEng = ThaiToEng(message)
    # memories = fetch_memories(OpenaiEmbedding(messageEng), load_convo() ,3)
    block = ''
    # for mem in memories:
    #     block += '%s\n' % (mem['message'])
    # block = block.strip()
    
    for doc in docs:
        block += 'From %s: %s\n' % (doc.metadata['source'], doc.page_content)
    block = block.strip()
    
    # Note: you need to be using OpenAI Python v0.27.0 for the code below to work
    # print(messageEng)
    print(block)
    

    
    response = GPTCompletions(messageEng, block).choices[0].message["content"]
    # decoded_str = response
    
    
    
    # for match in re.findall(utf16_pattern, response.encode()):
    #     decoded_str = decoded_str.replace(match.decode('utf-16'), '')

    print(translator.translate(response, dest=language).text)


