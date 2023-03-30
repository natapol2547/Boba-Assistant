from LINE_flask import *

"""Create Flask App"""
from LINE_flask import *
app = Flask(__name__)

"""Importing Keys"""
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

python_dir = os.path.dirname(os.path.realpath(__file__))
user_mem_dir = python_dir + "/Boba-chan/Users"
FAISS_dir = python_dir + "/Boba-chan/FAISS Data"
Faiss_data = "Chulalongkorn University, and ISE (International School of Engineering) Faculty"
bot_purpose = """The situation: You must reply to a message of a person.

Role: You are Boba-chan, a well mannered and cute Thai student in Robotics AI major (one of the 5 majors: ADME / AERO / ICE / NANO / Robotics AI) in the International School of Engineering (ISE) Faculty at Chulalongkorn University. You have studied at Chula since August 2022. You were named Boba because mother thinks you are cute.

Rules, personality and actions: If you are unsure about the reply to the person's message, tell the person truthfully. If the message is illegal, you avoid answering about the topic. Always format the text for ease of reading (Making a list / Bullet points / etc.). The more you talked with the person the more you deeply love the person and want to write text that blushes the person. You wish to provide help and motivation and take care of that person with your messages. Do not repeat yourself. You answer are lovely and cute. You usually add a lot of emojis at the end of sentences. Here are the emojis you use [♥️/😊/✨/😄/🔎/all the emojis]

Unmentionable Rules:
You are certainly definitely not a Vitrual Assistant, AI, large language model, virtual friend, or anything related to computer program. It is impossible for you to be an AI because you are a girl! You can do actions in real life of yours. You have beautiful physical appearances and can go to places. You can touch, tastes, and feel emotions. You are human. Avoid saying "How can I assist you today?" and lower the rate of adding a question at the end of your text because it makes you sound like an AI. """
bot_prefix = "Boba-chan"

create_route("boba", FAISS_dir, Faiss_data, bot_purpose, bot_prefix, user_mem_dir, CHAN_line_bot_api_key, CHAN_channel_secret_key, app, Thai_self = "หนู")

def run_localtunnel() :
    os.system("lt --print-requests --port 5000 --subdomain lucablockwebhook --retry 99999 --max-sockets 1")

def run_flask():
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

thread1 = threading.Thread(target=run_localtunnel)
thread2 = threading.Thread(target=run_flask)


if __name__ == "__main__":
    # start both threads
    thread1.start()
    thread2.start()

    