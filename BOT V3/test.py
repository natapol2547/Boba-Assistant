from langchain.agents import load_tools, initialize_agent, ZeroShotAgent, Tool, AgentExecutor
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.chains import RetrievalQA, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks import get_openai_callback
import time

import re
import requests
from typing import List, Dict, Union
from bs4 import BeautifulSoup

import os
from dotenv import load_dotenv
load_dotenv()
openai_key = os.getenv('OPENAI_KEY')
ASSISTANT_line_bot_api_key = os.getenv('ASSISTANT_LINEBOT_API_KEY')
ASSISTANT_channel_secret_key = os.getenv('ASSISTANT_CHANNEL_SECRET')

CHAN_line_bot_api_key = os.getenv('CHAN_LINEBOT_API_KEY')
CHAN_channel_secret_key = os.getenv('CHAN_CHANNEL_SECRET')

python_dir = os.path.dirname(os.path.realpath(__file__))
user_mem_dir = python_dir + "/Boba-chan/Users"
FAISS_dir = python_dir + "/Boba-chan/FAISS Data"
Faiss_data = "Chulalongkorn University and it's faculties especially ISE (International School of Engineering)"

def image_message(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    width, height = img.size
    gcd_value = gcd(width, height)
    aspect_ratio_str = f"{int(width / gcd_value)}:{int(height / gcd_value)}"
    # size = "full"
    return BubbleContainer(size = "kilo", hero=ImageComponent(url = url, size = "full", aspect_ratio=aspect_ratio_str, aspect_mode="cover", action=URIAction(uri=url)))