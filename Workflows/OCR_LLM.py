from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import uvicorn
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Eden AI Workflow API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Eden AI Configuration
EDEN_AI_TOKEN = os.getenv("EDEN_AI_API_KEY")
WORKFLOW_ID = "c9bce822-95db-4303-a955-e75dfbb9a9fc"
EDEN_AI_BASE_URL = "https://api.edenai.run/v2/workflow"

def get_headers():
    return {"Authorization": f"Bearer {EDEN_AI_TOKEN}"}

@app.post("/launch-execution/")
async def launch_execution(
    input_image: UploadFile = File(...),
    query: str = Form(...)
):
    """
    Launch a workflow execution with the Eden AI API
    """
    try:
        # Prepare the API request
        url = f"{EDEN_AI_BASE_URL}/{WORKFLOW_ID}/execution/"
        
        # Need to save the file temporarily to handle it properly
        temp_file_path = f"temp_{input_image.filename}"
        with open(temp_file_path, "wb") as buffer:
            contents = await input_image.read()
            buffer.write(contents)
        
        # Open file, make request, and ensure file is closed properly
        with open(temp_file_path, 'rb') as file_object:
            files = {"InputImage": file_object}
            payload = {"query": query}
            
            # Make the request to Eden AI
            response = requests.post(
                url, 
                files=files, 
                data=payload, 
                headers=get_headers()
            )
        
        # Now that the file is properly closed, try to remove it
        try:
            os.remove(temp_file_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {e}")
        
        # Return the response
        if response.status_code in [200, 201]:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Eden AI API error: {response.text}"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-execution-result/{execution_id}")
async def get_execution_result(execution_id: str):
    """
    Get the result of a workflow execution
    """
    try:
        url = f"{EDEN_AI_BASE_URL}/{WORKFLOW_ID}/execution/{execution_id}/"
        
        response = requests.get(url, headers=get_headers())
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Eden AI API error: {response.text}"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    Root endpoint that confirms the server is running
    """
    return {"message": "Eden AI Workflow API is running!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)