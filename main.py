import os
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import json
from typing import List

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
    
# --- Pydantic Models for a STRUCTURED Output ---

class LibraryExplanation(BaseModel):
    name: str
    explanation: str

class FunctionExplanation(BaseModel):
    name: str
    purpose: str
    inputs: str
    outputs: str

class FullExplanation(BaseModel):
    project_summary: str
    libraries: list[LibraryExplanation]
    functions: list[FunctionExplanation]
    execution_steps: str

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Code Mentor API!"}



# ... (keep all your existing code from the top of the file down to the end of the Pydantic models) ...

@app.post("/analyze-code", response_model=FullExplanation)
async def analyze_code(code_input: CodeInput):
    """
    Receives a piece of code and returns a structured JSON explanation from Gemini.
    """
    model = genai.GenerativeModel('gemini-2.5-pro')

    # This is our new, sophisticated prompt!
    prompt = f"""
    You are an expert AI code analyst. Your task is to analyze the provided Python code
    and generate a structured JSON output based on the schema I provide.
    Do not output anything other than the JSON object itself.

    **JSON Schema to follow:**
    {{
      "project_summary": "A high-level, one-paragraph explanation of what this code does.",
      "libraries": [
        {{
          "name": "library_name",
          "explanation": "A brief explanation of the library's role in this specific code and why it's a common choice."
        }}
      ],
      "functions": [
        {{
          "name": "function_name",
          "purpose": "A clear explanation of what this function does.",
          "inputs": "Description of the function's parameters and expected inputs.",
          "outputs": "Description of what the function returns."
        }}
      ],
      "execution_steps": "A simple, beginner-friendly guide on how to run this code (e.g., 'uvicorn main:app --reload')."
    }}

    **Python Code to Analyze:**
    ```python
    {code_input.code}
    ```
    """

    response = model.generate_content(prompt)

    # The response from Gemini is a string, so we need to parse it into a Python dictionary.
    # We also need to clean it up in case Gemini adds markdown backticks.
    cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
    try:
        structured_response = json.loads(cleaned_response)
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON from Gemini response."}

    return structured_response