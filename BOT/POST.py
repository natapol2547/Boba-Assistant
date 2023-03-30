import requests
import json

url = 'https://5e78-2405-9800-b670-2fe0-a0eb-39a1-33b8-a2ce.ap.ngrok.io/myapi'
data = {'name': 'John'}

response = requests.post(url, json=data)
print(response.text)
