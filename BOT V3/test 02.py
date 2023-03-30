import requests
from bs4 import BeautifulSoup
import re

def load_text_content(url):
    # Set options to ignore certificate verification and set timeout
    requests.packages.urllib3.disable_warnings()
    options = {
        'verify': False,
        'timeout': 5
    }

    # Make HTTP GET request to URL with options
    response = requests.get(url, **options)

    # Parse HTML content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract text content from HTML using BeautifulSoup's get_text() method
    text_content = soup.get_text()

    # Replace multiple consecutive spaces with a single space
    text_content = re.sub('\s+', ' ', text_content)

    # Replace multiple consecutive newlines with a single newline
    text_content = re.sub('\n+', '\n', text_content)

    return text_content


from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import traceback
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
def get_url_content(string):
    print(f"Loading Content: {string}")
    try:
        question,url = string.split(",")
    except:
        return "Wrong format of Action Input used. Use `<question>,<url>` format."
    try:
        # question_embedding = embeddings(question)
        text_splitter = CharacterTextSplitter(        
            separator = " ",
            chunk_size = 500,
            chunk_overlap  = 100,
            length_function = len,
        )

        extracted_text = load_text_content(url)
        # print(extracted_text)
        docs = text_splitter.create_documents([extracted_text], metadatas=[{"source": url}])
        # print(docs)
        db = FAISS.from_documents(docs, embeddings)
        related_docs = db.similarity_search(question, k=1)
        answer = ""
        for doc in related_docs:
            answer += doc.page_content + "\n\n"
        return f"Content of {url}: {answer} \n\n-------------\n\nAnalyze the content."
    except Exception as e: 
        print(traceback.format_exc())
        return f"Unable to load url content of {url}. Please try another url."

url = 'https://docs.midjourney.com/docs/plans'
text_content = get_url_content(f"Subscription plans,{url}")
print(text_content)

