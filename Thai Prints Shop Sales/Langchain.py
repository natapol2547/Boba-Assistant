from langchain.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from tqdm import tqdm

from googletrans import Translator #pip install googletrans==3.0.1a0
translator = Translator()




def ThaiToEng(x):
    return translator.translate(x, dest='en').text if x != '' else ''

def EngToThai(x):
    return translator.translate(x, dest='th').text if x != '' else ''


python_dir = os.path.dirname(os.path.realpath(__file__))

loader = DirectoryLoader(python_dir + '/Documents/', glob='**/*.txt')
data = loader.load()

print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your document')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
texts = text_splitter.split_documents(data)
print (f'Now you have {len(texts)} documents')

# print(texts[0])
# while 1:
#     pass

# print("Translating texts to English")
# for i in tqdm(range(len(texts))):
#     texts[i] = ThaiToEng(texts[i])

from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = 'us-east1-gcp'



embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "boba-assistant-vectors"
docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
query = "ผมสูบบุหรี่ได้ที่ไหน"
docs = docsearch.similarity_search(query, include_metadata=True)
print(docs)

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")
while True:
    query = ThaiToEng(input("User: "))
    docs = docsearch.similarity_search(query, include_metadata=True)
    print(EngToThai(chain.run(input_documents=docs, question=query)))