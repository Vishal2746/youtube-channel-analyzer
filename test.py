import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Test the Gemini API
try:
    model = genai.GenerativeModel('gemini-1.0-pro')
    response = model.generate_content("Hello, Gemini!")
    print(response.text)
except Exception as e:
    print("API key test failed:", e)