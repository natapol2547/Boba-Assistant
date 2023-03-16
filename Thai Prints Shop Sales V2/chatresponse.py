import os
import tiktoken

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



""" Langchain Chat """

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
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

from time import time, sleep

from uuid import uuid4

from langchain.embeddings.openai import OpenAIEmbeddings

def LUCABlockGetResponse(query, bot_purpose, FAISS_dir, users_mem_dir, user_id, webhookEventId, openai_key, newcomers_condition = "" ,load_message_count = 4 ,temperature = 0, bot_prefix = "AI"):
    conversation = load_convo(users_mem_dir, user_id)
    pastwebhookEventId =[]
    if conversation:
        for element in conversation:
            pastwebhookEventId.append(element["webhookEventId"])
    # print(pastwebhookEventId)
    
    # print(str(message).lower().startswith("boba"))
    if webhookEventId not in pastwebhookEventId:
        
        os.environ['OPENAI_API_KEY'] = openai_key
        embeddings = OpenAIEmbeddings()

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
        
        memories = get_last_messages(conversation, load_message_count)
        normal_chat = ChatOpenAI(temperature=0)
        
        
        time_now = time()
        # print(memories[-1:]['message_ai'])
        if memories and len(memories) > 0 and len(memories[-1]) > 0:
            special_condition = f"You have been talking to this person for {time_difference(conversation[0]['time'], time_now)}. Time for the earliest message of today: {time_of_earliest_message_of_today(conversation)}"
            message_salience = [
                SystemMessage(content="You are a conversation analyzer, meaning you are not to answer user's message. You are to determine what the user's intention is in 10 words. The following is a conversation between 2 people"),
                AIMessage(content=memories[-1]['message_ai']),
                HumanMessage(content=query + " In 15 words, user's intent is")
            ]
        else: 
            special_condition = newcomers_condition
            message_salience = [
                SystemMessage(content="You are a conversation analyzer, meaning you are not to answer user's message. You are to determine what the user's intention is in 10 words. The following is a conversation between 2 people"),
                HumanMessage(content=query + " In 15 words, user's intent is")
            ]
        user_salience = normal_chat(message_salience)
        user_message_with_salience = f"{query}.\nThough: The person is {user_salience.content}"
        messageVector = OpenaiEmbedding(user_message_with_salience)
        vectorstore = FAISS.load_local(FAISS_dir, embeddings)
        memories = fetch_memories(messageVector, conversation, 3) + memories





        template = """{bot_purpose}

        {context}
        
        {special_condition} 
        Answer as {bot_prefix}. The date and time currently is: {time}.
        
        Conversation:
        {chat_history}
        Human: {input}
        {bot_prefix}:"""

        prompt = PromptTemplate(
            input_variables=["bot_purpose", "special_condition", "time", "chat_history", "input", "context", "bot_prefix"], 
            template=template
        )
        
        """Remove duplicates"""
        memories = remove_duplicate_memory(memories)
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", ai_prefix= bot_prefix)
        for mem in memories:
            # if mem['speaker'] == "USER" :
            memory.save_context({"input": mem['message_user']}, {"output": mem['message_ai']})
            # else:
            #     memory.save_context()
        chain = load_qa_chain(ChatOpenAI(temperature=temperature, model_kwargs = {"presence_penalty": 1, "frequency_penalty": 1} , max_tokens = 1000), chain_type="stuff", memory=memory, prompt=prompt)
        # query = query
        docs = vectorstore.similarity_search_by_vector(messageVector, k=2)
        
        
        
        
        # bot_purpose = bot_purpose
        # chain({"input_documents": docs, "input": query}, return_only_outputs=True)
        # {'output_text': ' Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.'}
        
        response = chain({"input_documents": docs, "special_condition": special_condition ,"time": timestamp_to_datetime(time_now),  "bot_purpose" : bot_purpose,  "input": query, "bot_prefix" : bot_prefix}, return_only_outputs=True)['output_text']
        print(chain.memory.buffer)
        # print(timestamp_to_datetime(time_now))
        
        
        
        
        
        
        
        # """MANNNN"""
        # time_input = timestamp_to_datetime(time_now)
        
        # context = ''
        # for doc in docs:
        #     context += 'From %s :\n%s \n\n' % (doc.metadata['source'], doc.page_content)
        # context = context.strip()
        
        # messages = [HumanMessage(content=f"""{bot_purpose} 

        # {context}
        
        # {special_condition} 
        # Answer as {bot_prefix}. The date and time currently is: {time_input}.
        # """)]
        
        

        # for mem in memories:
        #     messages.extend([
        #     HumanMessage(content=mem["message_user"]),
        #     HumanMessage(content=mem["message_ai"])
        #     ])
        # messages.extend([
        #     HumanMessage(content=user_message_with_salience) 
        #     ])

        # # print(messages)
        # res = normal_chat(messages)
        # # print(f"Length: {len(res.content)}\n{res.content}")
        
        # response = res.content

        
        
        info = {'time': time(), 'vector': messageVector, 'message_user': user_message_with_salience, 'message_ai': response, 'webhookEventId': webhookEventId, 'uuid': str(uuid4())}
        filename = 'log_%s.json' % time()
        save_json(users_mem_dir + '/%s/%s' % (user_id, filename), info)
        
        
        
        
        
        return response
    else:
        print("--Duplicated webhook--")
        return "Duplicated Event"