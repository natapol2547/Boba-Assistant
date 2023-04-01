from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

python_dir = os.path.dirname(os.path.realpath(__file__))

from langchain.embeddings import HuggingFaceInstructEmbeddings
model_name = "hkunlp/instructor-large"
embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)


if input("Vectorize Documents? (Y/N): ").lower() == "y":

    """Determine the Documents Directory"""
    loader = DirectoryLoader(python_dir + '/Documents/', glob='**/*.txt')

    documents = loader.load()
    
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(docs)
    

    vectorstore = FAISS.from_documents(docs, embeddings)

    """Saving Vector Data"""
    vectorstore.save_local(python_dir + "/FAISS Data/")




# """ Langchain Chat """

# from langchain.chat_models import ChatOpenAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory

# vectorstore = FAISS.load_local(python_dir + "/FAISS Data/", embeddings)

# template = """{bot_purpose}

# {context}

# {chat_history}
# Human: {input}
# Chatbot:"""

# prompt = PromptTemplate(
#     input_variables=["bot_purpose", "chat_history", "input", "context"], 
#     template=template
# )
# memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")
# memory.save_context({"input": "Hello my name is Tawan"}, {"ouput": "Hello Tawan. How can I help you?"})
# chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt)
# query = "What is my name?"

# docs = vectorstore.similarity_search(query)
# bot_purpose = """You are a chatbot having a conversation with a human.

# Given the following extracted parts of a long document and a question, create a final answer."""
# # chain({"input_documents": docs, "input": query}, return_only_outputs=True)
# # {'output_text': ' Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.'}

# print(chain({"input_documents": docs, "bot_purpose" : bot_purpose,  "input": query}, return_only_outputs=True)['output_text'])