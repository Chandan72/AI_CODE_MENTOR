import os
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create an instance of the FastAPI application
app = FastAPI()

# --- Pydantic Model for Input ---
# This defines the structure of the data we expect in the request body.
class CodeInput(BaseModel):
    code: str

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Code Mentor API!"}

@app.post("/analyze-code")
async def analyze_code(code_input: CodeInput):
    """
    Receives a piece of code and returns a simple explanation from Gemini.
    """
    # Select the Gemini model
    model = genai.GenerativeModel('gemini-2.5-pro') # Or 'gemini-1.5-pro'

    # Create the prompt
    prompt = f"""
    You are an expert AI code assistant for beginner developers.
    Explain the following Python code in a clear, concise, and easy-to-understand way.
    Focus on the input, output, and logic of the code.

    Code:
    ```python
    {code_input.code}
    ```
    """

    # Generate content using the model
    response = model.generate_content(prompt)

    # Return the generated text
    return {"explanation": response.text}