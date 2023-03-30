from langchain.document_loaders.csv import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader

import os
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

python_dir = os.path.dirname(os.path.realpath(__file__))

embeddings = OpenAIEmbeddings()

python_dir = os.path.dirname(os.path.realpath(__file__))

loader = CSVLoader(file_path=python_dir + '/Documents/Sale Data collecting ENG 02.csv')

data = loader.load()
print(data)

vectorstore = FAISS.from_documents(data, embeddings)

"""Saving Vector Data"""
vectorstore.save_local(python_dir + "/FAISS Data/")