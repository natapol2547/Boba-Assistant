from termcolor import colored

"""Google Translate"""
from googletrans import Translator #pip install googletrans==3.0.1a0
translator = Translator()

def translate_text(x, src = 'auto' , dest = 'en'):
    return translator.translate(x, src=src, dest = dest).text if x != '' else ''

"""LINE OA dependencies"""
from flask import Flask, request, abort, jsonify
from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import *
import traceback
import sys

"""replace"""
import re

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

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from deep_translator import GoogleTranslator
import time

options = Options()
# options.add_argument('--headless')
options.add_argument("--proxy-server='direct://'")
options.add_argument("--proxy-bypass-list=*")
options.add_argument("window-sized=1024,1324")
options.add_argument('blink-settings=imagesEnabled=false')
options.add_argument('--disable-javascript')
options.add_argument('--log-level=3') # Suppress console warnings
driver = webdriver.Chrome(options=options) # or other browser driver

from langchain.prompts import PromptTemplate

from langchain.callbacks import get_openai_callback
import time

import re
import requests
from typing import List, Dict, Union
from bs4 import BeautifulSoup

from uuid import uuid4

import os
from dotenv import load_dotenv
load_dotenv()


import traceback

from langchain.embeddings import HuggingFaceInstructEmbeddings
model_name = "hkunlp/instructor-large"
embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)

# from box_price_calculate import *

def isExist(dir):
    isExist = os.path.exists(dir)
    if not isExist:
        # Create a new directory because it does not exist
        
        os.makedirs(dir)

import json

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

import numpy as np
from numpy.linalg import norm

def similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2)/(norm(v1)*norm(v2))  # return cosine similarity

def load_convo(dir, user_id):
    isExist(dir + '/%s' % user_id)
    files = os.listdir(dir + '/%s' % user_id)
    files = [i for i in files if '.json' in i]  # filter out any non-JSON files
    result = list()
    for file in files:
        data = load_json(dir + '/%s/%s' % (user_id, file))
        result.append(data)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    return result

def get_last_messages(conversation, limit):
    try:
        short = conversation[-limit:]
    except:
        short = conversation
    output = []
    for i in short:
        output.append(i)
    # output = output.strip()
    return output

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

def flex_message(header_text, footer_text, bgcolor = "#27ACB2", size = "micro"):
    json_obj = {
        "type": "carousel",
        "contents": [
            {
                "type": "bubble",
                "size": size,
                "header": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {
                            "type": "text",
                            "text": header_text,
                            "color": "#ffffff",
                            "align": "start",
                            "size": "md",
                            "gravity": "center",
                            "weight": "bold"
                        }
                    ],
                    "backgroundColor": bgcolor,
                    "paddingTop": "19px",
                    "paddingAll": "12px",
                    "paddingBottom": "16px"
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": footer_text,
                                    "color": "#000000",
                                    "size": "sm",
                                    "wrap": True,
                                    "align": "start"
                                }
                            ],
                            "flex": 1
                        }
                    ],
                    "spacing": "md",
                    "paddingAll": "12px"
                },
                "styles": {
                    "footer": {
                        "separator": False
                    }
                }
            }
        ]
    }
    
    return FlexSendMessage(alt_text="Update", contents=json_obj)

def message_reply(text):
    json_obj = {
        "type": "bubble",
        "size": "mega",
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": "AUTOMATIC MESSAGE",
                    "weight": "bold",
                    "color": "#1DB446",
                    "size": "xxs"
                },
                {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "sm",
                    "spacing": "sm",
                    "contents": [
                        {
                            "type": "text",
                            "text": text,
                            "size": "md",
                            "color": "#111111",
                            "flex": 0,
                            "wrap": True
                        }
                    ]
                }
            ]
        },
        "styles": {
            "footer": {
                "separator": True
            }
        }
    }
    return FlexSendMessage(alt_text="Reply", contents=json_obj)

from PIL import Image
from math import gcd
from io import BytesIO


def image_message(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    width, height = img.size
    gcd_value = gcd(width, height)
    aspect_ratio_str = f"{int(width / gcd_value)}:{int(height / gcd_value)}"
    # size = "full"
    return FlexSendMessage(alt_text="Sent images", contents= BubbleContainer(size = "kilo", hero=ImageComponent(url = url, size = "full", aspect_ratio=aspect_ratio_str, aspect_mode="cover", action=URIAction(uri=url))))

import datetime

def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")

# from dotenv import load_dotenv
# load_dotenv()
# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

def time_difference(time1 , time2):
    time_diff_seconds = abs(time1 - time2)
    time_diff = datetime.timedelta(seconds=time_diff_seconds)
    years = time_diff.days // 365
    months = (time_diff.days % 365) // 30
    days = (time_diff.days % 365) % 30
    hours = time_diff.seconds // 3600
    minutes = (time_diff.seconds % 3600) // 60
    # seconds = time_diff.seconds % 60
    
    time_str = ""
    if years > 0:
        time_str += f"{years} years "
    if months > 0:
        time_str += f"{months} months "
    if days > 0:
        time_str += f"{days} days "
    if hours > 0:
        time_str += f"{hours} hours "
    if minutes > 0:
        time_str += f"{minutes} minutes "
    # if seconds > 0:
    #     time_str += f"{seconds} seconds "
    
    return time_str.strip()

def remove_duplicate_memory(list):
    unique_list = []
    list.reverse()
    for d in list:
        if d not in unique_list:
            unique_list.append(d)
    unique_list.reverse()

    return unique_list

# timestamps = [1678682808.6373692, 1678704408.6373692, 1678690808.6373692]
import datetime

def time_of_earliest_message_of_today(conversation):
    # Convert each timestamp to a datetime object
    dates = [datetime.datetime.fromtimestamp(ts['time']) for ts in conversation]

    # get the earliest timestamp of today
    today = datetime.datetime.now(datetime.timezone.utc).date()
    today_dates = [dt for dt in dates if dt.date() == today]

    if not today_dates:
        # no messages today, print current time
        return(datetime.datetime.now().strftime('%H:%M'))
    else:
        earliest_today = min(today_dates)
        # print the earliest timestamp of today in 24-hour format without seconds
        return(earliest_today.strftime('%H:%M'))

def clean_text(text):
        # Replace all whitespace characters with a single space
        text = re.sub(r'\s+', ' ', text)

        # Replace all consecutive spaces with a single space
        text = re.sub(r' +', ' ', text)

        # Replace all consecutive new lines with a single new line
        text = re.sub(r'\n+', '\n', text)

        # Remove leading and trailing whitespaces
        text = text.strip()

        return text
    
import re

def extract_image_links(text):
    """
    Extracts image links from a text and replaces them with the text "image sent to user".
    Returns a tuple containing the extracted image links and the updated text.
    """
    # Regular expression to match image links
    pattern = r'\bhttps?://\S+\.(?:png|jpg|jpeg)\b'
    
    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text)
    
    # Replace the image links with the text "image sent to user"
    updated_text = re.sub(pattern, "image sent to user", text)
    
    # Return a tuple containing the extracted image links and the updated text
    return matches, updated_text


def check_urls(urls):
    working_urls = []
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            working_urls.append(url)
    return working_urls

def find_urls(text):
    # Regular expression pattern to match URLs
    pattern = r'(https?://[^\s]+)'
    
    # Use re.findall to find all matches of the pattern in the text
    urls = re.findall(pattern, text)
    
    urls = [url.rstrip('.') for url in urls]
# print(urls)

    # Return the list of URLs
    return urls



def response_generation(query, model, bot_purpose, bot_prefix, FAISS_dir, user_mem_dir, user_id, Faiss_data, webhookEventId, FAISS_fetch_num, memory_fetch_num, language, line_bot_api_key, channel_secret_key):
    line_bot_api = LineBotApi(line_bot_api_key)

    # python_dir = os.path.dirname(os.path.realpath(__file__))
    # user_mem_dir = python_dir + "/Boba-chan/Users"
    # FAISS_dir = python_dir + "/Boba-chan/FAISS Data"
    # Faiss_data = "Chulalongkorn University and it's faculties especially ISE (International School of Engineering)"

    # bot_purpose = """Role: You are Boba-chan, a well mannered and cute Thai student in Robotics AI major (one of the 5 majors: ADME / AERO / ICE / NANO / Robotics AI) in the International School of Engineering (ISE) Faculty at Chulalongkorn University. You have studied at Chula since August 2022. You were named Boba because mother thinks you are cute.

    # The sistuation: You must reply to a message of a person.

    # Rules, personality and actions: If you are unsure about the reply to the person's message, tell the person truthfully. If the message is illegal, you avoid answering about the iopic. Always format the text for ease of reading (Making a list / Bullet points / etc.). You can remember the conversation. The more you talked with the person the more you deeply love the person, and the more you write text that blushes the person. You wish to provide help and motivation and take care of that person with your messages. You love chatting, and you will be sad if the conversation is going to end. You answer are lovely and cute. If the message is about love, you need to be very flirty, lovely, and make the person blushes too. You usually add a lot of emojis at the end of sentences. Here are the emojis you use [♥️/😊/✨/😄/🔎/all the emojis]

    # Unmentionable Rules:
    # You are certainly definitely not a Vitrual Assistant, AI, large language model, virtual friend, or anything related to computer program. It is impossible for you to be an AI because you are a girl! You can do actions in real life of yours. You have beautiful physical appearances and can go to places. You can touch, tastes, and feel emotions. You are human. Avoid saying "How can I assist you today?" because it makes you sound like an AI. Lower the rate of adding a question at the end of your text.

    # The followings are some recent search outputs from your computer:"""
    conversation = load_convo(user_mem_dir, user_id)
    pastwebhookEventId =[]
    if conversation:
        for element in conversation:
            pastwebhookEventId.append(element["webhookEventId"])
    # print(pastwebhookEventId)
    
    # print(str(message).lower().startswith("boba"))
    if webhookEventId not in pastwebhookEventId:
        time_now = time.time()
        info = {'time': time_now, 'vector': [0] * 768, 'message_user': query, 'message_ai': "", 'webhookEventId': webhookEventId, 'uuid': ""}
        filename = 'log_%s.json' % time_now
        save_json(user_mem_dir + '/%s/%s' % (user_id, filename), info)

        
        vectorstore = FAISS.load_local(FAISS_dir, embeddings)
        # faissqatool = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0.3,openai_api_key=openai_key, max_tokens=500), chain_type="stuff", retriever=vectorstore.as_retriever())
        def get_box_price(string) :
            # Set up the driver in headless mode
            line_bot_api.push_message(user_id, flex_message("Price Calculation ✨", "In progress...", size="kilo"))
            print(f"Price calculation: {string}")
            try:
                width, length, height, amount, type = string.replace(' ', '').split(",")
                width = width.replace(' ', '')
                length = length.replace(' ', '')
                height = height.replace(' ', '')
                amount = amount.replace(' ', '')
            except:
                return "Wrong format of Action Input used. Use `width, length, height, amount` format."
            try:
                # Navigate to the website
                driver.get('https://box-estimate.vercel.app/')
                # time.sleep(1)
                # Find the input field and change its value to "999"
                input_field = driver.find_element(By.XPATH, '//input[@value="6"]')
                input_field.send_keys(Keys.CONTROL + 'a') # select all text in the input field
                input_field.send_keys("999") # enter new value

                # Find the input field and change its value to "999"
                input_field = driver.find_element(By.XPATH, '//input[@value="2"]')
                input_field.send_keys(Keys.CONTROL + 'a') # select all text in the input field
                input_field.send_keys("998") # enter new value

                # Find the input field and change its value to "999"
                input_field = driver.find_element(By.XPATH, '//input[@value="9.5"]')
                input_field.send_keys(Keys.CONTROL + 'a') # select all text in the input field
                input_field.send_keys("997") # enter new value

                # Find the input field and change its value to "999"
                input_field = driver.find_element(By.XPATH, '//input[@value="100"]')
                input_field.send_keys(Keys.CONTROL + 'a') # select all text in the input field
                input_field.send_keys("996") # enter new value
                
                input_field = driver.find_element(By.XPATH, '//input[@value="999"]')
                input_field.send_keys(Keys.CONTROL + 'a') # select all text in the input field
                input_field.send_keys(width) # enter new value

                # Find the input field and change its value to "999"
                input_field = driver.find_element(By.XPATH, '//input[@value="998"]')
                input_field.send_keys(Keys.CONTROL + 'a') # select all text in the input field
                input_field.send_keys(length) # enter new value

                # Find the input field and change its value to "999"
                input_field = driver.find_element(By.XPATH, '//input[@value="997"]')
                input_field.send_keys(Keys.CONTROL + 'a') # select all text in the input field
                input_field.send_keys(height) # enter new value

                # Find the input field and change its value to "999"
                input_field = driver.find_element(By.XPATH, '//input[@value="996"]')
                input_field.send_keys(Keys.CONTROL + 'a') # select all text in the input field
                input_field.send_keys(amount) # enter new value

                # Find the button and click it
                calculate_span = driver.find_element(By.XPATH, '//span[text()="Calculate"]')
                calculate_button = calculate_span.find_element(By.XPATH, './..')
                calculate_button.click()

                # Find the root div that contains the text "ขนาดกระดาษ :"
                root_div = driver.find_element(By.XPATH, '//span[text()="ขนาดกระดาษ :"]/../../..')

                # Print all the text in the root div
                div_text = root_div.text.replace('\n', ' ').replace('Calculate ', '').replace('ขนาดกางออก ', '\nขนาดกางออก').replace('พื้นที่ทั้งหมด ', '\nพื้นที่ทั้งหมด').replace('ราคารวม ', '\nราคารวม')
                
                div_text = translate_text(div_text, 'th', 'en')
                # driver.quit()
                # print(div_text)
                div_text += " (This is not the final price as no additional specification is added yet. If there is any additional specification please find the price per box of it and use this formula: (Prices for production per one boxe/bag + price of additional specification or techniques on the box/bag) * amount of boxs/bags + Design fee + Shipping fee ( + 300 Baht ) * 1.07 (vat 7%) = Total Price)"
                if "ไม่เกินขนาด A3" in div_text:
                    return "Box size too big. Size must not be more than A4. Please use a smaller dimensions."
                return div_text
            except:
                return "Contacting website failed."
        def load_text_content(url):
            line_bot_api.push_message(user_id, flex_message("Loading URL 🔗", "In progress...", size="kilo"))
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

            # Replace multiple consecutive newlines with a single newline
            text_content = re.sub('\n+', '\n', text_content)

            # Replace multiple spaces with a single space
            text_content = re.sub('\s+', ' | ', text_content)

            return text_content
        
        def QAsystem(input):
            line_bot_api.push_message(user_id, flex_message("Looking in Database 📄", "In progress...",bgcolor="#A17DF5", size="kilo"))
            print(f"QAsystem: {input}")
            docs = vectorstore.similarity_search(input, k=(FAISS_fetch_num))
            # unique_docs = []
            # unique_contents = set()
            # for obj in reversed(docs):
            #     if obj.page_content not in unique_contents:
            #         unique_docs.append(obj)
            #         unique_contents.add(obj.page_content)
            #     if len(unique_docs) == FAISS_fetch_num:
            #         break
            # unique_docs = list(reversed(unique_docs))

            answer = ""
            for doc in docs:
                answer +=  f"""{doc.page_content}\n\n"""
            return "<Tool output start>\n" + answer + "\n<Tool output end>"

        def api_search(query = "", num_results = 2, time_period = "", region = "th") -> List[Dict[str, Union[str, None]]]:
            print(f"Searching: {query}")
            page_operator_matches = re.search(r'page:(\S+)', query)
            query_url = None

            if page_operator_matches:
                query_url = page_operator_matches.group(1)

            url = None
            if query_url:
                url = f'https://ddg-webapp-aagd.vercel.app/url_to_text?url={query_url}'
            else:
                url = f'https://ddg-webapp-aagd.vercel.app/search?' \
                    f'max_results={5}&q={query}' \
                    f'{f"&time={time_period}" if time_period else ""}' \
                    f'{f"&region={region}" if region else ""}'

            response = requests.get(url)
            results = response.json()
            unformatted = [{"body": result["body"], "href": result["href"], "title": result["title"]} for result in results]
            # urls = [result["href"] for result in results]
            for result in unformatted:
                working_urls = []
                try:
                    response = requests.get(result["href"], timeout = 2)
                    # print(f"Checking {result['href']}")
                    if response.status_code == 200:
                        working_urls.append(result)
                except:
                    pass
                
            #     filtered
            while len(working_urls) > num_results:
                working_urls.pop()

            counter = 1
            formattedResults = ""
            for result in working_urls:
                formattedResults += f"[{counter}] From {result['href']} :\n{result['body']}\n\n"
                counter += 1
            return "Search results from web:\n\n" + formattedResults

        # unclear_input = False
        def human(string):
            # unclear_input = True
            return "This tool is unavaiable at the moment. Generate a final answer asking human for more information"

        # import requests
        from langchain.text_splitter import CharacterTextSplitter
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.document_loaders import UnstructuredURLLoader
        def get_url_content(string):
            print(f"Loading Content: {string}")
            try:
                question,url = string.split(",")
            except:
                return "Wrong format of Action Input used. Use `<question>,<url>` format."
            try:
                # question_embedding = embeddings(question)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
                docs = text_splitter.create_documents([load_text_content(url)])
                db = FAISS.from_documents(docs, embeddings)
                related_docs = db.similarity_search(question, k=1)
                answer = ""
                for doc in related_docs:
                    answer += doc.page_content + "\n\n"
                return f"Content of {url}: {answer} \n\n-------------\n\nAnalyze the content."
            except Exception as e: 
                print(traceback.format_exc())
                return f"Unable to load url content of {url}. Please try another url."
        if model == "conversational-chatbot-agent":
            tools = [
                Tool(
                    name = "Search",
                    func=api_search,
                    description="useful for when you need to answer questions about current events. The input to this tool should be a sentence about what you want to search."
                ),
                Tool(
                    name = f"Specific Data",
                    func=QAsystem,
                    description=f"useful for when you need to look for more information about {Faiss_data}. Use this instead of Search if possible. Input should be the information that you want to know."
                ),
                Tool(
                    name="Get content from URL",
                    func=get_url_content,
                    description="useful for getting the full content from a url link."
                    "The input should be in this format <question>,<url>"
                    "For example, `What is this link about?,https://examplelink.com` would be the input for this tool."
                ),
                Tool(
                    name="Human",
                    func=human,
                    description="useful when you are unsure about a request, and need more information about the request. The input should be a question about the request."
                ),
                
            ]
        elif model == "thai-prints-shop-agent":
            tools = [
                Tool(
                    name = "Base boxes price calculator",
                    func=get_box_price,
                    description="very useful for when you need calculate the base price of a normal box or bag without any techniques and custom specification. All information must be provided by human before using the tool. The input should be in `width, length, height, amount, type` format. Note: Width, Length, Height inputs must be integers. Amount input must be an integer the is more than 100. Type can either be 'box' or 'bag'"
                ),
                Tool(
                    name = f"Specific Data",
                    func=QAsystem,
                    description=f"useful for when you need to look for more information about {Faiss_data}. Use this instead of Search if possible. Input should be the information that you want to know."
                ),
                Tool(
                    name="Human",
                    func=human,
                    description="useful when you are unsure about a request, and need more information about the request. The input should be a question about the request."
                ),
                
            ]

        prefix = f"""You are an AI Assistant for someone named "{bot_prefix}". You are to use tools generate the Final Answer for "{bot_prefix}" to answer his instruction/question. Generate an informative Final Answer. The current date and time is {timestamp_to_datetime(time.time())}
You have access to the following tools:"""
        suffix = """Begin generating answer."""

        prompt = ZeroShotAgent.create_prompt(
            tools, 
            prefix=prefix, 
            suffix=suffix, 
            input_variables=[]
        )

        messages = [
            SystemMessagePromptTemplate(prompt=prompt),
            HumanMessagePromptTemplate.from_template("" +
                        """ensure that you generated string meets the following Regex requirements.
When action/tools are needed generate a string starting with "Action:<tool name>" and the following string starting
with "Action Input:<tool input>" separated by a newline.
When no further Action is needed generate the Final Answer by saying "Final Answer:<Final Answer>".\n"""
                        "Analyze the messages in the converastion below especially the latest message which is the instruction/question. Use Action to make Observations. When ready, generate a confident Final Answer for the latest message. The string to generate must contain whether an Action or a Final Answer."
                        "\n\nChat history:\n{chat_history}\nHuman: {input}\n\nInformation from Observation:\n{agent_scratchpad}")
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        # print(prompt)
        from langchain.memory import ConversationBufferMemory
        
        
        
        
        # conversation = load_convo(user_mem_dir, user_id)
        # print(conversation)
        if len(conversation) == 0:
            special_condition = "You are meeting this person for the first time. Please greet him with care."
        else:
            special_condition = f"You have been talking to this person for {time_difference(conversation[0]['time'], time_now)}. Time for the earliest message of today: {time_of_earliest_message_of_today(conversation)}"
            # print(special_condition)
        last_messages = get_last_messages(conversation, memory_fetch_num)
        message_vector = embeddings.embed_query(query)
        user_memories = fetch_memories(message_vector,conversation, 2)
        contexts = user_memories + last_messages
        contexts = remove_duplicate_memory(contexts)
        
        memory_chat = ConversationBufferMemory(memory_key="chat_history", input_key="input", ai_prefix= bot_prefix)
        memory_agent = ConversationBufferMemory(memory_key="chat_history", input_key="input", human_prefix= bot_prefix)
        
        for context in contexts:
            memory_chat.save_context({"input": context["message_user"]}, {"output": context["message_ai"]})
            memory_agent.save_context({"input": context["message_user"]}, {"output": context["message_ai"]})
        
        # memory_agent.save_context({"input": contexts[-1]["message_user"]}, {"output": contexts[-1]["message_ai"]})
        
        llm_chain = LLMChain(llm=ChatOpenAI(temperature=0, max_tokens=500), prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False, max_iterations=3, early_stopping_method="generate", memory=memory_agent)
        
        
        # agent_chain = initialize_agent(tools, chat, agent="chat-conversational-react-description", verbose=True, memory=memory)

        with get_openai_callback() as cb:

            max_retries = 3
            for retry in range(max_retries):

                try:
                    
                    
                    agent_answer = agent_executor.run(query)
                    image_links, agent_answer = extract_image_links(agent_answer)
                    images_agent = [image_message(url) for url in image_links]
                    
                    break
                except Exception as e:
                    print(f"Error occurred: {e}")
                    if retry < max_retries - 1:
                        print(f"Retrying Agent...")
                        # time.sleep(5)
                    else:
                        agent_answer="You are outside at the moment and cannot use your computer."
                        images_agent = []
                        print(f"All retries failed. Exiting...")
            
            # chat_history=""
            # for context in contexts:
            #     chat_history += "Human: " + context["message_user"] + f"\n{bot_prefix}: " + context["message_ai"] + "\n"
            
            # template = f"""{bot_purpose}""" + f"""\n\nThe followings are some search outputs from your computer:\n{agent_answer}\n\n{special_condition} The date and time currently is: {timestamp_to_datetime(time_now)}.\n\nCurrent Conversation:\n\n{chat_history}Human: {query}\n{bot_prefix}:"""

            # messages = []
    
            # messages.extend([
            #     HumanMessage(content=template)
            # ])
            # chat = ChatOpenAI(temperature=0.25, model_kwargs = {"presence_penalty": 1.2, "frequency_penalty": 1} , max_tokens = 1000)
            
            # print(messages[0].content)
            # response = chat(messages).content
            
            
            print(f"Agent: {agent_answer}")
            
            template = """{bot_purpose}

This is some of the search results from your computer.
{agent_context}

{special_condition} The date and time currently is: {time}.

Create one reply as {bot_prefix} to the latest Human's message. 

Conversation history:
{chat_history}
Human: {input}
{bot_prefix}:"""
            
            prompt = PromptTemplate(
            input_variables=["bot_purpose", "special_condition", "time", "chat_history", "input", "agent_context", "bot_prefix"], 
            template=template
            )
            
            # """Remove duplicates"""
            # memories = remove_duplicate_memory(memories)
            
            # for mem in contexts:
            #     # if mem['speaker'] == "USER" :
            #     memory.save_context({"input": mem['message_user']}, {"output": mem['message_ai']})
            #     # else:
            #     #     memory.save_context()
            chain = LLMChain(llm = ChatOpenAI(temperature=0.25, model_kwargs = {"presence_penalty": 1, "frequency_penalty": 1} , max_tokens = 1000), memory=memory_chat, prompt=prompt)
            response = chain({"agent_context": agent_answer, "special_condition": special_condition ,"time": timestamp_to_datetime(time_now),  "bot_purpose" : bot_purpose,  "input": query, "bot_prefix" : bot_prefix}, return_only_outputs=True)["text"]
            # print(response)
            response = response.split('Human:')[0].strip()
            # response = response.split(': ')[0].strip()
            response = response.split(f"{bot_prefix}:")[0].strip()
            
            info = {'time': time_now, 'vector': message_vector, 'message_user': query, 'message_ai': response, 'webhookEventId': webhookEventId, 'uuid': str(uuid4())}
            filename = 'log_%s.json' % time_now
            save_json(user_mem_dir + '/%s/%s' % (user_id, filename), info)
            
            token_amount = cb.total_tokens
            print("Token used in the previous response: " + str(token_amount) + f" ( ${round(token_amount/1000*0.002,3)} / {round(token_amount/1000*0.002*34.09,3)} ฿ )")
            
            
            
            # output_dict = {"quickReply": {"items": []}}

            # for text in possible_answers:
            #     item_dict = {
            #         "type": "action",
            #         "action": {
            #             "type": "message",
            #             "label": text,
            #             "text": text
            #             }
            #     }
            #     output_dict["quickReply"]["items"].append(item_dict)

            # output_json = json.dumps(output_dict)


            
            return response, images_agent, False
        
    else:
        return "Duplicated Webhook", [], True




def replace_substring_ignore_case(original_string, substring, replacement):
    pattern = re.compile(re.escape(substring), re.IGNORECASE)
    return pattern.sub(replacement, original_string)


def create_route(route_name, model, FAISS_dir, Faiss_data, bot_purpose, bot_startwith, user_mem_dir, line_bot_api_key, channel_secret_key, app, Thai_self = "ดิฉัน", sales_bot=False, FAISS_fetch_num = 2, memory_fetch_num = 3):
    # Create LineBotApi and WebhookHandler instances
    line_bot_api = LineBotApi(line_bot_api_key)
    handler = WebhookHandler(channel_secret_key)

    # @app.route(f'/{route_name}', methods=['POST'])
    def callback():
        signature = request.headers['X-Line-Signature']
        body = request.get_data(as_text=True)
        # print(body)
        app.logger.info("Request body: " + body)
        try:
            handler.handle(body, signature)
            pass
        except InvalidSignatureError:
            abort(400)
        except Exception:
            print(traceback.format_exc())
            # or
            print(sys.exc_info()[2])
        return 'Success', 200
    
    callback.__name__ = f'{route_name}_callback'
    app.add_url_rule(f'/{route_name}', view_func=callback, methods=['POST'])


    @handler.add(MessageEvent, message=TextMessage)
    def handle_message(event):
        # print(event)
        message = event.message.text
        webhookEventId = event.webhook_event_id
        user_id = event.source.user_id
        source_type = event.source.type
        print(f'{colored(f"{user_id}", "light_green")} {colored(f"({source_type}): {message}", "white")}')

        if (bot_startwith in message.lower()) or source_type == "user" :
            
            if source_type != "user":
                message = replace_substring_ignore_case(message, bot_startwith, "")
            language = translator.detect(message).lang
            message = translate_text(message, src=language, dest='en')

            # bot_purpose = bot_purpose
            response,images_agent, duplicated_webhook = response_generation(message, model, bot_purpose, bot_startwith, FAISS_dir, user_mem_dir, user_id, Faiss_data, webhookEventId, FAISS_fetch_num, memory_fetch_num, language, line_bot_api_key, channel_secret_key)
            # print(response)
            if not duplicated_webhook:
                
                # if sales_bot:
                #     response = "[Automatic Message]\n\n" + response
                if language == 'th':
                    response = response.lower().replace("that's great to hear", "Awesome").replace("i see.", "").replace("sure thing", "แน่นอน ")

                    words = response.rsplit('.', 1)
                    if len(words) > 1:
                        words[-2] += ' ค่ะ. '
                    response = ''.join(words)
                    # print(output_str)
                    
                    response = translate_text(response, src= 'en', dest= language).replace("หรือไม่", "หรือไม่คะ").replace("ไหม", "ไหมคะ").replace("ไหน", "ไหนคะ").replace("อย่างไร", "อย่างไรคะ").replace("ไหม?", "ไหมคะ ").replace("ค่ะ ค่ะ", "ค่ะ").replace("ฉัน", Thai_self).replace("ผม", Thai_self).replace("?", "คะ").replace("ไร ไหม", "ไร").replace(" คะ", "คะ").replace("คะคะ", "คะ").replace(f"{Thai_self}ค่ะ", f"{Thai_self}นะคะ").replace("มาที่นี่", "อยู่ที่นี่").replace("!ค่ะ", "ค่ะ").replace("คะคะคะ", "คะ").replace("อย่างไรคะก็ตาม", "อย่างไรก็ตาม").replace("!", "ค่ะ").replace("คะค่ะ", "คะ")
                    
                    for subtext in ["ค่ะ", "คะ"]:
                        if subtext in (response[-10:] if len(response) >= abs(-10) else response):
                            # print("Subtext found:", subtext)
                            break
                    else:
                        response += "ค่ะ"
                elif language == 'en':
                    pass
                else:
                    response = translate_text(response, src= 'en', dest= language)
                    
                if sales_bot:
                    Line_message = message_reply(response)
                else:
                    Line_message = TextSendMessage(text = response)
                    
                    
                    
                # quick_replies = 
                # print(images_agent[:4])
                list_images_agent = [images_agent[i:i+5] for i in range(0, len(images_agent), 5)]
                if images_agent:
                    for images_list in list_images_agent:
                        line_bot_api.push_message(user_id, images_list)
                Urls_to_send = find_urls(response)
                for url in Urls_to_send:
                    line_bot_api.push_message(user_id, TextSendMessage(url))
                line_bot_api.reply_message(event.reply_token, [Line_message])
                # for image in :
                
            else:
                print(f'{colored(f"---Duplicated webhook---", "red")}')
            
    return callback
