import requests
import json

url = 'https://e7bd-2405-9800-b670-2fe0-8196-69e-20e-d33.ap.ngrok.io/myapi'
data = {'name': 'John'}

response = requests.post(url, json=data)
print(response.text)
