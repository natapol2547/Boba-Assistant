from agent import *
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
from PIL import Image
from math import gcd
from io import BytesIO
"""replace"""
import re

def image_message(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    width, height = img.size
    gcd_value = gcd(width, height)
    aspect_ratio_str = f"{int(width / gcd_value)}:{int(height / gcd_value)}"
    # size = "full"
    return FlexSendMessage(BubbleContainer(size = "kilo", hero=ImageComponent(url = url, size = "full", aspect_ratio=aspect_ratio_str, aspect_mode="cover", action=URIAction(uri=url))))

print(image_message("https://python.langchain.com/en/latest/modules/agents/agents/custom_agent.html"))