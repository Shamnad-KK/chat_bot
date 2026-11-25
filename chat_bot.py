from dotenv import load_dotenv

# os is used for accessing loaded environment variables
import os
from openai import OpenAI

from transformers import AutoTokenizer
import time


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_base = os.getenv("GROQ_API_BASE_URL")

if groq_api_key is None:
    raise ValueError(
        "GROQ_API_KEY not found in environment variables. Please set it in the .env file.")

if groq_api_base is None:
    raise ValueError(
        "GROQ_API_BASE not found in environment variables. Please set it in the .env file.")

class ChatBot:
    def __init__(self):
        self.prompts = []
        self.total_tokens = 0
        self.client = OpenAI(api_key=groq_api_key, base_url=groq_api_base)

    def bot(self):
        # response = self.client.chat.completions.create(model="openai/gpt-oss-20b", messages= self.prompts)
        # content = response.choices[0].message.content
        streamedResponse = self.client.chat.completions.create(model="openai/gpt-oss-20b", 
        messages= self.prompts, stream=True)

        content = ""
        for chunk in streamedResponse:
            if chunk.choices:
                if chunk.choices[0].delta:
                    if chunk.choices[0].delta.content:
                        # print(chunk.choices[0].delta.content, end='', flush=True)
                        # time.sleep(0.05) 
                        content += chunk.choices[0].delta.content
        print()  # Print a newline after the response is complete
        if chunk.usage:
            print('hi')
            self.total_tokens += chunk.usage.total_tokens
        self.prompts.append({"role": "assistant", "content": content})
        return content

    def ask(self, prompt):
        self.prompts.append({"role": "user", "content": prompt})
        response = self.bot()
        return response

    def reset(self):
        self.prompts.clear()
        print("Chat history cleared. You can start a new conversation.")

    def chat(self):
        while True:
            prompt = input('Ask something: ')
            encoding = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
            token_count = len(encoding.encode(prompt))
            print(f'Prompt Token count: {token_count}')

            limit = 10

            if self.total_tokens > 400:
                print(f"Your prompt exceeds the maximum token limit of today. Please shorten your input.")
                break

            if token_count > limit:
                print(f"Your prompt exceeds the maximum token limit of {limit}. Please shorten your input.")
                continue

            if prompt.lower() == 'exit':
                self.prompts.clear()
                print("Exiting the chat. Goodbye!")
                break

            elif prompt.lower() == 'reset':
                self.reset()
                continue

            response = self.ask(prompt)

            print(f'Bot: {response}')
            print(f'Total Tokens Used: {self.total_tokens}')

if __name__ == "__main__":
    chat_bot = ChatBot()
    chat_bot.chat()
