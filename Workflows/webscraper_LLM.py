from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from typing import Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
EDEN_AI_API_TOKEN = os.getenv("EDEN_AI_API_KEY")
EDEN_AI_WORKFLOW_URL = "https://api.edenai.run/v2/workflow/53247399-2f86-47ab-8a1b-73d16e4abf12/execution/"

@app.post("/process-file/")
async def process_file(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    try:
        # Read file content
        file_content = await file.read()
        
        # Prepare the headers
        headers = {
            "Authorization": f"Bearer {EDEN_AI_API_TOKEN}"
        }
        
        # Prepare the files and payload
        files = {
            "file": (file.filename, file_content, file.content_type)
        }
        
        payload = {
            "text": question
        }
        
        # Make the request to Eden AI
        response = requests.post(
            EDEN_AI_WORKFLOW_URL,
            files=files,
            data=payload,
            headers=headers
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Return the response from Eden AI
        result = response.json()
        
        # If the execution succeeded, process the OCR results
        if result["content"]["status"] == "succeeded":
            try:
                ocr_results = result["content"]["results"]["ocr__ocr_async"]["results"][0]
                
                # Extract text from all pages
                text = ""
                for page in ocr_results["pages"]:
                    for line in page["lines"]:
                        text += line["text"] + "\n"
                
                return {
                    "status": "succeeded",
                    "text": text.strip()
                }
            except KeyError:
                # If the structure is different than expected
                return result
        
        return result
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error communicating with Eden AI: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

@app.get("/status/{execution_id}")
async def get_execution_status(execution_id: str):
    try:
        headers = {
            "Authorization": f"Bearer {EDEN_AI_API_TOKEN}"
        }
        
        # Make request to Eden AI to get current status
        url = f"{EDEN_AI_WORKFLOW_URL}{execution_id}/"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error communicating with Eden AI: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)