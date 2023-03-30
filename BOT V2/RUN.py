"""Load Keys"""
import os
from dotenv import load_dotenv
from chatresponse import *


load_dotenv()
openai_key = os.getenv('OPENAI_KEY')


"""LINE OA dependencies"""
from flask import Flask, request, abort, jsonify
from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import *
app = Flask(__name__)

"""Google Translate"""
from googletrans import Translator #pip install googletrans==3.0.1a0
translator = Translator()

def translate_text(x, src = 'auto' , dest = 'en'):
    return translator.translate(x, src=src, dest = dest).text if x != '' else ''

"""Start app"""
chula_line_bot_api = LineBotApi(os.getenv('CHAN_LINEBOT_API_KEY'))
chula_handler = WebhookHandler(os.getenv('CHAN_CHANNEL_SECRET'))
@app.route("/chula", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    # print(body)
    app.logger.info("Request body: " + body)
    try:
        chula_handler.handle(body, signature)
        pass
    except Exception as e:
        print(e)
        abort(400)
    return 'Success', 200

@chula_handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message = event.message.text
    webhookEventId = event.webhook_event_id
    user_id = event.source.user_id
    print(f"{user_id}: {message}")
    
    if "boba" in message.lower():
        message = message.lower().replace("boba", "")
        language = translator.detect(message).lang
        message = translate_text(message, src=language, dest='en')
        
        python_dir = os.path.dirname(os.path.realpath(__file__))
        user_mem_dir = python_dir + "/Users"
        FAISS_dir = python_dir + "/FAISS Data"

        bot_purpose = """You are a chatbot having a conversation with an emplyee of a company. The employee will ask you questions about the company rules. 

        Given the following extracted parts of a long document (the company rules) and a question, create a informative final answer. If you cannot answer the question with the provided information, say truthfully that you don't know the answer."""
        response = LUCABlockGetResponse(message, bot_purpose, FAISS_dir, user_mem_dir, user_id, webhookEventId, openai_key)
        print(response)
        if response != "Duplicated Event":
            response = translate_text(response, src= 'en', dest= language)
            if language == 'th':
                response = response.replace("?", "คะ").replace("ฉัน", "ดิฉัน")
                if not (response.endswith("ค่ะ") or response.endswith("คะ")):
                    response += "ค่ะ"
            chula_line_bot_api.reply_message(event.reply_token, TextSendMessage(response))





if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)