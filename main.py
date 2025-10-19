import os
import zipfile
from fastapi import UploadFile, File
# Make sure you also have these from before

import google.generativeai as genai
from openai import OpenAI

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import json
from typing import List
import git
import tempfile
import shutil
from fastapi import HTTPException
import stat
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()


# Configure the Gemini API client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
client=OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("QWEN_API_KEY")
    
)

# Create an instance of the FastAPI application
app = FastAPI()


# --- ADD THIS MIDDLEWARE SECTION ---
# This is the crucial part for allowing the frontend to talk to the backend.
origins = [
    "http://localhost:3000", # The origin of our Next.js frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)
# --- END OF MIDDLEWARE SECTION ---



# --- Pydantic Model for Input ---
# This defines the structure of the data we expect in the request body.
class CodeInput(BaseModel):
    code: str
    
class RepoInput(BaseModel):
    github_url: str
    
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
    architecture_diagram: str

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
      "execution_steps": "A simple, beginner-friendly guide on how to run this code (e.g., 'uvicorn main:app --reload').",
      "architecture_diagram": "A simple flowchart diagram in Mermaid.js syntax that shows the high-level execution flow or component interaction(end goal is to build the great mental model such user can develop very good understanding of what is going under the hood). For example: graph TD; A[User Input] --> B(Backend Analysis); B --> C{{Gemini API}}; C --> D[Display Result];"
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


# Add this helper function to your file
def handle_remove_readonly(func, path, exc_info):
    """
    Error handler for shutil.rmtree.

    If the error is due to a read-only file, it changes the permissions
    and retries the deletion.
    """
    # Check if the error is a PermissionError
    if not os.access(path, os.W_OK):
        # Change the file to be writable
        os.chmod(path, stat.S_IWRITE)
        # Retry the function that failed (e.g., os.unlink)
        func(path)
    else:
        raise


@app.post("/analyze-repo", response_model=FullExplanation)
async def analyze_repo(repo_input: RepoInput):
    """
    Clones a public GitHub repository, analyzes its content, and returns a structured explanation.
    """
    temp_dir = tempfile.mkdtemp() # Create a secure, temporary directory

    try:
        # 1. Clone the repository
        try:
            git.Repo.clone_from(repo_input.github_url, temp_dir)
        except git.GitCommandError:
            # This handles invalid URLs or private repositories
            shutil.rmtree(temp_dir, onerror=handle_remove_readonly)
            raise HTTPException(status_code=400, detail="Could not clone repository. Check if the URL is correct and the repository is public.")

        # 2. Consolidate relevant code files into a single context string
        consolidated_context = ""
        ignore_list = ['.git', '__pycache__', 'venv', '.vscode']
        max_files = 50  # Safety limit to avoid processing huge repos
        file_count = 0

        for root, dirs, files in os.walk(temp_dir):
            # Modify the dir list in-place to prevent os.walk from descending into ignored folders
            dirs[:] = [d for d in dirs if d not in ignore_list]

            for file in files:
                if file_count >= max_files:
                    break
                # We focus on common file types for now
                if file.endswith(('.py', '.md', '.txt')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            consolidated_context += f"--- FILE: {os.path.relpath(file_path, temp_dir)} ---\n"
                            consolidated_context += content + "\n\n"
                            file_count += 1
                    except Exception:
                        # Ignore files that can't be read (e.g., binary files)
                        continue
            if file_count >= max_files:
                break

        if not consolidated_context:
             raise HTTPException(status_code=400, detail="No readable files found in the repository.")

        # 3. Use the same sophisticated prompt, but with the full repo context
        
        prompt = f"""
        You are an expert AI code analyst. Your task is to analyze the provided codebase from a full project repository
        and generate a structured JSON output based on the schema I provide. The project's files and their content are
        concatenated below. Do not output anything other than the JSON object itself.

        **JSON Schema to follow:**
        {{
          "project_summary": "A high-level, one-paragraph explanation of what this entire project does. Infer this from the README.md and the code.",
          "libraries": [
            {{
              "name": "library_name",
              "explanation": "Identify key libraries from requirements.txt or import statements. Explain their role in this specific project and why they are common choices."
            }}
          ],
          "functions": [
            {{
              "name": "function_name (include file path if multiple files)",
              "purpose": "A clear explanation of what this function does.",
              "inputs": "Description of the function's parameters and expected inputs.",
              "outputs": "Description of what the function returns."
            }}
          ],
          "execution_steps": "Based on the files, provide a simple, beginner-friendly guide on how to set up and run this project.",
          "architecture_diagram": "A simple flowchart diagram in Mermaid.js syntax that shows the high-level execution flow. For example: 'graph TD; A[User Input] --> B(Analysis); B --> C{{Gemini API}};'. **If no logical diagram can be made for this code, you MUST return null for this field.**"

        **Full Project Codebase:**
        ```
        {consolidated_context}
        ```
        """
        # --- Call OpenRouter using OpenAI format ---
        
         # Or {code_input.code} for the /analyze-code endpoint

        try:
            chat_completion = client.chat.completions.create(
              model="qwen/qwen3-coder", # <-- Make sure this is the correct model name from OpenRouter
              response_format={ "type": "json_object" }, # Ask for JSON directly
              messages=[
                {
                  "role": "system",
                  "content": "You are an AI assistant that analyzes code and ONLY responds with valid JSON matching the user's requested schema."
                },
                {
                  "role": "user",
                  "content": prompt, # Use the prompt content defined above
                },
              ],
            )

            # Extract the JSON string from the response
            response_text = chat_completion.choices[0].message.content

            # Parse the JSON string
            # No need to manually clean '```json' if response_format works
            structured_response = json.loads(response_text)
            return structured_response

        except Exception as e:
            # Keep your existing error handling
            if "ResourceExhausted" in str(e) or "429" in str(e): # Check for rate limit errors
                 raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait a minute and try again.")
            print(f"Error during OpenRouter API call: {e}") # Print error for debugging
            raise HTTPException(status_code=503, detail=f"The AI service failed to process the request. Error: {e}")
        

        response = model.generate_content(prompt)

        # 4. Clean and parse the response
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        try:
            structured_response = json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to decode JSON from Gemini response.")

        return structured_response

    finally:
        # 5. Clean up: ALWAYS remove the temporary directory
        shutil.rmtree(temp_dir, onerror=handle_remove_readonly)
        

@app.post("/analyze-zip", response_model=FullExplanation)
async def analyze_zip(file: UploadFile = File(...)):
    """
    Receives a .zip file, extracts it, analyzes its content, and returns a structured explanation.
    """
    # Create a temporary directory to extract files into
    extract_dir = tempfile.mkdtemp()
    temp_zip_path=None

    try:
        # --- THIS BLOCK IS MODIFIED ---
        # 1. Create a temp file but use delete=False
        #    This means the file will NOT be deleted when the 'with' block ends.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
            shutil.copyfileobj(file.file, temp_zip)
            temp_zip_path = temp_zip.name  # 2. Store the file's path

        # 3. NOW, the 'with' block is closed. The file lock is released.
        #    We can safely open the file at temp_zip_path to read it.
        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid .zip file.")

        # --- REUSE THE REPO ANALYSIS LOGIC ---
        consolidated_context = ""
        ignore_list = ['__pycache__', '.vscode'] # We don't ignore .git here
        max_files = 50
        file_count = 0

        for root, dirs, files in os.walk(extract_dir):
            dirs[:] = [d for d in dirs if d not in ignore_list]
            for file in files:
                if file_count >= max_files: break
                if file.endswith(('.py', '.md', '.txt')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            consolidated_context += f"--- FILE: {os.path.relpath(file_path, extract_dir)} ---\n"
                            consolidated_context += content + "\n\n"
                            file_count += 1
                    except Exception:
                        continue
            if file_count >= max_files: break

        if not consolidated_context:
             raise HTTPException(status_code=400, detail="No readable .py, .md, or .txt files found in the .zip.")

        # --- END OF REUSED LOGIC ---

        # Call Gemini (same as before)
        model = genai.GenerativeModel('gemini-2.5-pro')
        prompt = f"""
        You are an expert AI code analyst. Your task is to analyze the provided codebase from a full project repository
        and generate a structured JSON output based on the schema I provide. The project's files and their content are
        concatenated below. Do not output anything other than the JSON object itself.

        **JSON Schema to follow:**
        {{
          "project_summary": "A high-level, one-paragraph explanation of what this entire project does. Infer this from the README.md and the code.",
          "libraries": [
            {{
              "name": "library_name",
              "explanation": "Identify key libraries from requirements.txt or import statements. Explain their role in this specific project and why they are common choices."
            }}
          ],
          "functions": [
            {{
              "name": "function_name (include file path if multiple files)",
              "purpose": "A clear explanation of what this function does.",
              "inputs": "Description of the function's parameters and expected inputs.",
              "outputs": "Description of what the function returns."
            }}
          ],
          "execution_steps": "Based on the files, provide a simple, beginner-friendly guide on how to set up and run this project.",
          "architecture_diagram": "A simple flowchart diagram in Mermaid.js syntax that shows the high-level execution flow. For example: 'graph TD; A[User Input] --> B(Analysis); B --> C{{Gemini API}};'. **If no logical diagram can be made for this code, you MUST return null for this field.**"

        **Full Project Codebase:**
        ```
        {consolidated_context}
        ```
        """

        try:
            response = model.generate_content(prompt)
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            structured_response = json.loads(cleaned_response)
            return structured_response
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"The AI service failed to process the request. Error: {e}")

    finally:
        # ALWAYS clean up the temporary directory
        shutil.rmtree(extract_dir, onerror=handle_remove_readonly)