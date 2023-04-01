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

from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()

from box_price_calculate import *

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

def check_urls(urls):
    working_urls = []
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            working_urls.append(url)
    return working_urls

def load_text_content(url):
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

def response_generation(query, model, bot_purpose, bot_prefix, FAISS_dir, user_mem_dir, user_id, Faiss_data, webhookEventId, FAISS_fetch_num, memory_fetch_num):
    
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


        def QAsystem(input):
            print(f"QAsystem: {input}")
            docs = vectorstore.similarity_search(input, k=FAISS_fetch_num)
            answer = ""
            for doc in docs:
                answer +=  f""""{doc.page_content}"\nURL: {doc.metadata["source"]}\n\n"""
            return answer

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
            return "Generate a final answer apologizing, and ask human for more information"

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
                    name = f"{Faiss_data} Direct Data",
                    func=QAsystem,
                    description=f"very useful for when you need to answer questions about {Faiss_data}. Use this instead of Search if possible. Input should be a clear question."
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
                    description="useful when you are unsure of the input, and need more information about it. The input should be a question about the input."
                ),
                
            ]
        elif model == "thai-prints-shop-agent":
            tools = [
                Tool(
                    name = "Box price calculator",
                    func=get_box_price,
                    description="useful for when you need calculate the accurate price of a box or bag. Do not hesitate to use the tool or guess the price of box yourself. The input should be in `width, length, height, amount, type` format. Note: Width, Length, Height are in centimeters. Type can either be 'box' or 'bag'"
                ),
                Tool(
                    name = f"{Faiss_data} Direct Data",
                    func=QAsystem,
                    description=f"very useful for when you need to answer questions about {Faiss_data}. Use this instead of Search if possible. Input should be a clear question."
                ),
                Tool(
                    name="Human",
                    func=human,
                    description="useful when you are unsure of the input, and need more information about it. The input should be a question about the input."
                ),
                
            ]

        prefix = f"""Role: You are an AI Assistant for a person named {bot_prefix}. You are to analyze and guide {bot_prefix} find answers and thought about the messages {bot_prefix} got from human. You always generate informative and long Final Answer with citations. The current date and time is {timestamp_to_datetime(time.time())}
Analyze the latest message mainly. You may not need tools in a normal conversation, although you have access to the following tools:"""
        suffix = """Begin generating answer."""

        prompt = ZeroShotAgent.create_prompt(
            tools, 
            prefix=prefix, 
            suffix=suffix, 
            input_variables=[]
        )

        messages = [
            SystemMessagePromptTemplate(prompt=prompt),
            HumanMessagePromptTemplate.from_template("Chat history:\n\n{chat_history}\n\n" +
                        "Analyze the text below. Gernerate the answer according to the given format."
                        """ensure that it meets the following Regex requirements.
                        The string starting with "Action:<action>" and the following string starting
                        with "Action Input:<action_input>" should be separated by a newline.
                        When no Action/Tool is needed, say "Thought:<AI thought>" and "Final Answer:<informative and long Final Answer>".\n\n"""
                        "{input}{agent_scratchpad}")
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
        memory_agent = ConversationBufferMemory(memory_key="chat_history", input_key="input", ai_prefix= bot_prefix)
        
        for context in contexts:
            memory_chat.save_context({"input": context["message_user"]}, {"output": context["message_ai"]})
            memory_agent.save_context({"input": context["message_user"]}, {"output": context["message_ai"]})
        
        # memory_agent.save_context({"input": contexts[-1]["message_user"]}, {"output": contexts[-1]["message_ai"]})
        
        llm_chain = LLMChain(llm=ChatOpenAI(temperature=0.15, max_tokens=300), prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False, max_iterations=2, early_stopping_method="generate", memory=memory_agent)
        
        
        # agent_chain = initialize_agent(tools, chat, agent="chat-conversational-react-description", verbose=True, memory=memory)

        with get_openai_callback() as cb:

            max_retries = 3
            for retry in range(max_retries):

                try:
                    
                    
                    agent_answer = agent_executor.run(query)
                    break
                except Exception as e:
                    print(f"Error occurred: {e}")
                    if retry < max_retries - 1:
                        print(f"Retrying Agent...")
                        # time.sleep(5)
                    else:
                        agent_answer="You are outside at the moment and cannot use your computer."
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
            
            chat = ChatOpenAI(temperature=0)
            messages = [
                HumanMessage(content=query),
                AIMessage(content=response),
                HumanMessage(content="Generate a numbered list of 5 of one word replies for Human from the conversation.")
            ]
            chat_possible_answer = chat(messages).content.split('\n')
            print(chat_possible_answer)
            # Remove the list numbers from each item in the output list
            try:
                possible_answers = [item.split('. ', 1)[1] for item in chat_possible_answer if item]
            except:
                possible_answers = ["My order", "More product", "Contact admin"]
            
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


            
            return response, possible_answers, False
        
    else:
        return "Duplicated Webhook", "", True
