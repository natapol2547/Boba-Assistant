from langchain.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from tqdm import tqdm
import json
import datetime
from time import time, sleep
from uuid import uuid4
import numpy as np
from numpy.linalg import norm
from googletrans import Translator #pip install googletrans==3.0.1a0
translator = Translator()

import os
import openai

import re

utf16_pattern = re.compile(b'\\xff\\xfe(.*?)\\x00')

def ThaiToEng(x):
    return translator.translate(x, dest='en').text if x != '' else ''

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

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')
openai.api_key = os.getenv('OPENAI_KEY')


from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# loader = DirectoryLoader(python_dir + '/Documents/', glob='**/*.txt')
# data = loader.load()

# print (f'You have {len(data)} document(s) in your data')
# print (f'There are {len(data[0].page_content)} characters in your document')

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# texts = text_splitter.split_documents(data)
# print (f'Now you have {len(texts)} documents')

def OpenaiEmbedding(prompt):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try: 
            vector = embeddings.embed_query(prompt)
            return vector
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(3)

# for idx, t in enumerate(tqdm(texts)):
#     content = t.page_content
#     message_vector = OpenaiEmbedding(t.page_content)

#     info = {'speaker': 'USER', 'index': idx, 'vector': message_vector, 'message': content , 'uuid': str(uuid4())}
#     filename = 'log_%s_USER.json' % idx
#     isExist(python_dir + '/Documents/JSON/')
#     save_json(python_dir + '/Documents/JSON/' + '%s' % (filename), info)



    
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
                {"role": "system", "content": f"""You are Boba AI, an AI Assistant of a company. You can only answer questions about the company. Answer the question as concisely as possible. Current date: {current_date} 
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

'''
Preparing LINE API connections 🔗
'''

from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *
#LineBot

line_bot_api = LineBotApi(os.getenv('LINEBOT_API_KEY'))
handler = WebhookHandler(os.getenv('CHANNET_SECRET'))

app = Flask(__name__)


# Local domain
@app.route("/company", methods=['POST'])
def callback():

    
    signature = request.headers['X-Line-Signature']

    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
        pass
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message = event.message.text
    print(message)
    user_id = event.source.user_id
    print(str(message).lower().startswith("boba"))
    if str(message).lower().startswith("boba"):
        message = message[4:]
        language = translator.detect(message).lang
        messageEng = translator.translate(message, dest='en').text
        memories = fetch_memories(OpenaiEmbedding(messageEng), load_convo() ,3)
        
        block = ''
        for mem in memories:
            block += '%s\n' % (mem['message'])
        block = block.strip()
        # print(block)
        response = translator.translate(GPTCompletions(messageEng, block).choices[0].message["content"], dest=language).text
        # print(response)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(response))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

