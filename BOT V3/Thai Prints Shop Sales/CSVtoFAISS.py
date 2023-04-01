from langchain.document_loaders.csv_loader import CSVLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from deep_translator import GoogleTranslator
def translate_string(string, source_language, target_language):
    # while True:
    #     try:
    result = GoogleTranslator(source=source_language, target=target_language).translate(string)
            # break
        # except:
        #     pass
    return result

import os
from dotenv import load_dotenv
import nest_asyncio
from tqdm import tqdm

nest_asyncio.apply()


load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

python_dir = os.path.dirname(os.path.realpath(__file__))

from langchain.embeddings import HuggingFaceInstructEmbeddings
model_name = "hkunlp/instructor-large"
embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)

python_dir = os.path.dirname(os.path.realpath(__file__))

loader = CSVLoader(file_path=python_dir + '/Documents/Sale Data collecting ENG 02.csv')

# data = loader.load()
data = []
loader2 = CSVLoader(file_path=python_dir + '/Documents/web.csv')
urls = [data.page_content.replace("Links: ", "") for data in loader2.load()]
print("Loading urls")
loader = WebBaseLoader(urls)
loader.requests_per_second = 10
data = loader.aload()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
data = text_splitter.split_documents(data)
for i in tqdm(range(len(data)), desc="Translating page content"):
    data[i].page_content = translate_string(data[i].page_content, "auto", "en")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
data = text_splitter.split_documents(data) 


print("Load completed!")

vectorstore = FAISS.from_documents(data, embeddings)

"""Saving Vector Data"""
vectorstore.save_local(python_dir + "/FAISS Data/")