from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import os
import threading
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')
ASSISTANT_line_bot_api_key = os.getenv('ASSISTANT_LINEBOT_API_KEY')
ASSISTANT_channel_secret_key = os.getenv('ASSISTANT_CHANNEL_SECRET')

CHAN_line_bot_api_key = os.getenv('CHAN_LINEBOT_API_KEY')
CHAN_channel_secret_key = os.getenv('CHAN_CHANNEL_SECRET')

SALES_line_bot_api_key = os.getenv('SALES_LINEBOT_API_KEY')
SALES_channel_secret_key = os.getenv('SALES_CHANNEL_SECRET')



chat = ChatOpenAI(temperature=0)
messages = [
    HumanMessage(content="What is the meaning of love?")
]
print(chat(messages))