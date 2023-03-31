from langchain.document_loaders.csv_loader import CSVLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


import os
from dotenv import load_dotenv
import nest_asyncio

nest_asyncio.apply()


load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

python_dir = os.path.dirname(os.path.realpath(__file__))

from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()

python_dir = os.path.dirname(os.path.realpath(__file__))

loader = CSVLoader(file_path=python_dir + '/Documents/Sale Data collecting ENG 02.csv')

data = loader.load()
# loader2 = CSVLoader(file_path=python_dir + '/Documents/web.csv')
# urls = [data.page_content.replace("Links: ", "") for data in loader2.load()]
# print("Loading urls")
# loader = WebBaseLoader(urls)
# loader.requests_per_second = 10
# data += loader.aload()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
# texts = text_splitter.split_documents(data)

print("Load completed!")

vectorstore = FAISS.from_documents(data, embeddings)

"""Saving Vector Data"""
vectorstore.save_local(python_dir + "/FAISS Data/")