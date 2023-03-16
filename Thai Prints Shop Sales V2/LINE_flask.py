from chatresponse import *

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

def replace_substring_ignore_case(original_string, substring, replacement):
    pattern = re.compile(re.escape(substring), re.IGNORECASE)
    return pattern.sub(replacement, original_string)


def create_route(route_name, FAISS_dir, bot_purpose, bot_startwith, user_mem_dir, line_bot_api_key, channel_secret_key , openai_key, app, Thai_self = "ดิฉัน", temperature = 0, newcomers_condition = "", load_message_count = 3):
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
        print(f"{user_id} ({source_type}): {message}")

        if (bot_startwith in message.lower()) or source_type == "user" :
            message = replace_substring_ignore_case(message, bot_startwith, "")
            language = translator.detect(message).lang
            message = translate_text(message, src=language, dest='en')

            # bot_purpose = bot_purpose
            response = LUCABlockGetResponse(query = message, bot_purpose= bot_purpose, FAISS_dir= FAISS_dir, users_mem_dir= user_mem_dir, user_id= user_id, webhookEventId= webhookEventId, openai_key= openai_key, temperature= temperature, newcomers_condition = newcomers_condition, bot_prefix=route_name, load_message_count=load_message_count)
            # print(response)
            if response != "Duplicated Event":
                if language == 'th':
                    response = response.lower().replace("that's great to hear", "Awesome").replace("i see.", "").replace("sure thing", "แน่นอน ").replace("?", "? คะ")

                    words = response.rsplit('.', 1)
                    if len(words) > 1:
                        words[-2] += ' ค่ะ. '
                    response = ''.join(words)
                    # print(output_str)
                    response = translate_text(response, src= 'en', dest= language).replace("ค่ะ ค่ะ", "ค่ะ").replace("ฉัน", Thai_self).replace("ผม", Thai_self).replace("?", "คะ").replace("ไร ไหม", "ไร").replace(" คะ", "คะ").replace("คะคะ", "คะ").replace(f"{Thai_self}ค่ะ", f"{Thai_self}นะคะ").replace("มาที่นี่", "อยู่ที่นี่")
                    
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
                
                
                    
                line_bot_api.reply_message(event.reply_token, TextSendMessage(response))

    return callback
