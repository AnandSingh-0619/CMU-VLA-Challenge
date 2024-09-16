import requests
import json
import os
from mistralai import Mistral
import time
class QueryVLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def query_gemini(self, query):
        api_key="Your API Key: DO NOT USE"
        url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}'
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": query
                        }
                    ]
                },
            ],
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    def query_chatgpt(self, query):
        api_key = "your_chatgpt_api_key"
        url = f'https://api.openai.com/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": query}],
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    def query_mistral(self, query):
        time.sleep(2)
        api_key="Your API Key: USE THIS"
        model = "mistral-large-latest"

        client = Mistral(api_key=api_key)

        chat_response = client.chat.complete(
            model= model,
            messages = [
                {
                    "role": "user",
                    "content": query,
                },
            ]
        )
        print(chat_response.choices[0].message.content)
        return chat_response.choices[0].message.content

    def query(self, query):
        
        if self.model_name == "gemini":
            return self.query_gemini(query)
        elif self.model_name == "chatgpt":
            return self.query_chatgpt(query)
        elif self.model_name == "mistral":
            return self.query_mistral(query)
        else:
            print("Model not supported")
            return None
