from fastapi import FastAPI

app=FastAPI()

@app.get("/")
def read_root():
    """
    this is the main endpoint which greet the user
    """
    return {"message": "Hello, World!"}