from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any, Union
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Eden AI API Integration", 
              description="Simplified FastAPI application for Eden AI text and image analysis")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment variables
EDEN_AI_API_KEY = os.getenv("EDEN_AI_API_KEY")
if not EDEN_AI_API_KEY:
    print("Warning: EDEN_AI_API_KEY not found in environment variables")

# Headers for Eden AI API
def get_headers():
    return {"Authorization": f"Bearer {EDEN_AI_API_KEY}"}

# ------ MODELS ------

class TextRequest(BaseModel):
    text: str
    model: str = "openai/gpt-4o"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class ImageUrlRequest(BaseModel):
    image_url: str
    prompt: str
    model: str = "openai/gpt-4o"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

# ------ ENDPOINTS ------

# Text generation endpoint
@app.post("/api/text", response_class=JSONResponse)
async def text_completion(request: TextRequest):
    """
    Generate text responses using Eden AI
    """
    try:
        url = "https://api.edenai.run/v2/llm/chat"
        
        # Create the proper structure for Eden AI
        payload = {
            "model": request.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": request.text
                        }
                    ]
                }
            ]
        }
        
        if request.temperature is not None:
            payload["temperature"] = request.temperature
            
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        
        response = requests.post(url, json=payload, headers=get_headers())
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Eden AI API Error: {response.text}"
            )
            
        return response.json()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Image analysis with URL endpoint
@app.post("/api/image", response_class=JSONResponse)
async def analyze_image(request: ImageUrlRequest):
    """
    Analyze an image from a URL using Eden AI
    """
    try:
        url = "https://api.edenai.run/v2/llm/chat"
        
        payload = {
            "model": request.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": request.prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": request.image_url
                            }
                        }
                    ]
                }
            ]
        }
        
        if request.temperature is not None:
            payload["temperature"] = request.temperature
            
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        
        response = requests.post(url, json=payload, headers=get_headers())
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Eden AI API Error: {response.text}"
            )
            
        return response.json()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)