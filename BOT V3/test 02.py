from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import *

quickreply = """{
    "items": [
      {
        "type": "action",
        "imageUrl": "https://example.com/sushi.png",
        "action": {
          "type": "message",
          "label": "Sushi",
          "text": "Sushi"
        }
      }"""
# print(QuickReplyButton(action=MessageAction("Hello", "Hello")))
print(TextSendMessage(text = "Hello", quick_reply=QuickReply(items=[QuickReplyButton(action=MessageAction("Hello", "Hello"))])))