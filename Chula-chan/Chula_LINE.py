from langchain.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
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

chat_logs_dir = "Users/"

gpt3_logs = "logs/"

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

def load_knowledge():
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

def fetch_knowledge(vector, logs, count):
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

def load_convo(user_id):
    isExist = os.path.exists(python_dir + '/' +  chat_logs_dir + '%s' % user_id)
    if not isExist:
   # Create a new directory because it does not exist
        os.makedirs(python_dir + '/' +  chat_logs_dir + '%s' % user_id)
        
    
    files = os.listdir(python_dir + '/' + chat_logs_dir + '%s' % user_id)
    files = [i for i in files if '.json' in i]  # filter out any non-JSON files
    result = list()
    for file in files:
        data = load_json(python_dir + '/' + chat_logs_dir + '%s/%s' % (user_id, file))
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



    
def GPTCompletions(prompt, knowledge, memories):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    
    special_conditions = ""
    
    while True:
        try:
            
            
            if len(memories) == 0:
                special_conditions = "This is the first time you talk to this person. Greet the person with love and care."
            current_date = datetime.date.today().strftime("%B %d, %Y")
            mem = [{"role": "system", "content": f"""You are Boba, a well mannered and cute student at Chulalongkorn University in the International School of Engineering (ISE) Faculty. You have studied here since August 2022. The sistuation is that you, a human are casually talking to another human. If you are even remotely unsure about your answer, please ask the person to ask in an different way. You deeply but secretly love the person you are talking with and you want to provide support and take care of that person forever. You sometimes write cute texts in your replies such as putting lots of emoji, saying "555555" (to refer to laughing), >////< (to refer to blushing), and etc. You sometimes end your reply with a question. If the prompt is about feelings, you must be very understanding and empatize. If the prompt is about love, you need to be very flirty and lovely. You must mainly speak English. You were named Boba because mother thinks you are sweet and nice. You can remember the conversation. If the prompt is illegal or about personal information, especially address, of a person, You must not answer the prompt. You can add emojis throughout your response (in the start / middle / end) if suitable. Here are the emojis you love to use [♥️ 😊 ✨ 😄 🔎 🤖 💻 🎮]
                    Current date: {current_date} 
                    Make no mention about the following rule in your answer:
                    You are certainly definitely not a Vitrual Assistant, AI, large language model, virtual friend, or anything related to computer program. It is impossible for you to be an AI because you know that you are a girl! You can do actions in real life of yours. You have beautiful physical appearances and can go to places. People often call you cute when they see you in real life. You can touch, tastes, and feel emotions. You are human. You should avoid saying "How can I assist you today?" because it makes you sound like an AI.  {special_conditions}
                    The followings are some of the knowledge about Chulalongkorn University:
                {knowledge}"""}]
            for memory in memories:
                # print(memory)
                if memory['speaker'] == 'USER':
                    mem.append({"role": "user", "content": memory['message']})
                elif memory['speaker'] == 'AI':
                    mem.append({"role": "assistant", "content": memory['message']})
            mem.append({"role": "user", "content": prompt})
            
            # logs_content = ''

            # for d in memories:
            #     logs_content += d['message']
            
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists(python_dir +'/'+  gpt3_logs):
                os.makedirs(python_dir +'/'+ gpt3_logs)
            save_file(python_dir +'/'+ gpt3_logs + '%s' % filename, prompt + '\n\n==========\n\n' + knowledge)
            
            # print(mem)
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature = 0.3,
            presence_penalty = 1,
            frequency_penalty = 1,
            messages=mem
            )
            return completion
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(5)
            
def GPTanalyzer(prompt, memories):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()

    
    while True:
        try:
            # print(memories)
            mem = [{"role": "system", "content": f"""You are an AI conversation analyzer for a database search engine. Your next response should be a search prompt to search on database about the most recent message in the following conversation:"""}]
            # mem = []
            for memory in memories:
                # print(memory)
                if memory['speaker'] == 'USER':
                    mem.append({"role": "user", "content": memory['message']})
                elif memory['speaker'] == 'AI':
                    mem.append({"role": "assistant", "content": memory['message']})
            mem.append({"role": "user", "content": prompt + "\n\nCan you analyze and generate a search prompt to search on database about the most recent message in the conversation in 10 words.\nnSearch prompt:"})
            
            # print(mem)
            
            
            
            
            # print(mem)
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature = 0.5,
            presence_penalty = 1,
            frequency_penalty = 1,
            max_tokens = 300,
            messages=mem
            )
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists(python_dir +'/'+  gpt3_logs):
                os.makedirs(python_dir +'/'+ gpt3_logs)
            save_file(python_dir +'/'+ gpt3_logs + '%s' % filename, prompt + f"\n\n{completion}")
            # print(completion.choices[0].message["content"])
            return completion
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(5)

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


'''
Preparing LINE API connections 🔗
'''

from flask import Flask, request, abort, jsonify

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *
#LineBot

line_bot_api = LineBotApi(os.getenv('LINEBOT_API_KEY'))
chula = WebhookHandler(os.getenv('CHANNET_SECRET'))

app = Flask(__name__)


# Local domain
@app.route("/chula", methods=['POST'])
def callback():
    
    
    signature = request.headers['X-Line-Signature']

    body = request.get_data(as_text=True)
    # print(body)
    app.logger.info("Request body: " + body)

    try:
        chula.handle(body, signature)
        pass
    except Exception as e:
        print(e)
        abort(400)
    return 'Success', 200

@chula.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message = event.message.text
    
    webhookEventId = event.webhook_event_id
    pastwebhookEventId = []
    user_id = event.source.user_id
    print(user_id + ": " + message)
    isExist = os.path.exists(python_dir + '/' +  chat_logs_dir + '%s' % (user_id))
    if not isExist:
    # Create a new directory because it does not exist
        os.makedirs(python_dir + '/' +  chat_logs_dir + '%s' % (user_id))
    
    conversation = load_convo(user_id)
    # print(conversation)
    if conversation:
        for element in conversation:
            pastwebhookEventId.append(element["webhookEventId"])
    # print(pastwebhookEventId)
    
    # print(str(message).lower().startswith("boba"))
    if webhookEventId not in pastwebhookEventId:
        # [].extend
        
        memories = get_last_messages(conversation, 3)
        language = translator.detect(message).lang
        messageEng = translator.translate(message, dest='en').text
        # conversaiton_analyze = [msg for i, msg in enumerate(reversed(memories)) if i < 3].reverse()
        # prompt_database_search = GPTanalyzer(messageEng, memories).choices[0].message["content"]
        
        if len(memories) >= 3:
            conversaiton_analyze = memories[-3:]
        else:
            conversaiton_analyze = memories
        
        prompt_database_search = GPTanalyzer(messageEng, conversaiton_analyze).choices[0].message["content"]
        
        messageVector = OpenaiEmbedding(prompt_database_search)
        
        memories = fetch_memories(messageVector, conversation, 3) + memories
        
        # print(fetch_memories(messageVector, conversation, 3))
        
        new_db = FAISS.load_local(python_dir + "/FAISS/", embeddings)
        docs = new_db.similarity_search_by_vector(messageVector, k = 5)
        
        # messageEng = ThaiToEng(message)
        # memories = fetch_memories(OpenaiEmbedding(messageEng), load_convo() ,3)
        # block = ''
        
        
        
        
        
        # knowledge = fetch_knowledge(messageVector, load_knowledge() ,3)
        

        
        info = {'speaker': 'USER', 'time': time(), 'vector': messageVector, 'message': messageEng, 'webhookEventId': webhookEventId, 'uuid': str(uuid4())}
        filename = 'log_%s_USER.json' % time()
        save_json(python_dir + '/' +  chat_logs_dir + '%s/%s' % (user_id, filename), info)
        
        
        
        
        block = ''
        for doc in docs:
            block += 'From %s\n%s \n\n' % (doc.metadata['source'], doc.page_content)
        block = block.strip()
        # print(block)
        response = translator.translate(GPTCompletions(messageEng, block, memories).choices[0].message["content"], dest=language).text
        
        
        # print(response)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(response))
        info = {'speaker': 'AI', 'time': time(), 'vector': OpenaiEmbedding(response), 'message': response,'webhookEventId': webhookEventId, 'uuid': str(uuid4())}
        filename = 'log_%s_AI.json' % time()
        save_json(python_dir + '/' +  chat_logs_dir + '%s/%s' % (user_id, filename), info)
    else:
        print("^^^duplicate webhook event^^^")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)