from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())
api_key = os.environ['OPENAI_API_KEY']
