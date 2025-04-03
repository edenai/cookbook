from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import os
from dotenv import load_dotenv
import requests

# Import our Eden AI adapter
from adapter import EdenAI

# Load environment variables
load_dotenv()

app = FastAPI(title="Eden AI OpenAI SDK API", 
              description="FastAPI application to use Eden AI with OpenAI SDK interface")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check for API key
EDEN_AI_API_KEY = os.getenv("EDEN_AI_API_KEY")
if not EDEN_AI_API_KEY:
    print("Warning: EDEN_AI_API_KEY not found in environment variables")

# Dependency to get client
def get_client():
    return EdenAI(api_key=EDEN_AI_API_KEY)

# Models for API requests
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatRequest(BaseModel):
    model: str = "openai/gpt-4o"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

# Text and image chat endpoint
@app.post("/chat/completions")
async def chat_completions(
    request: ChatRequest,
    client: EdenAI = Depends(get_client)
):
    """
    Process a chat completion request using Eden AI with OpenAI SDK interface
    """
    try:
        # Convert Pydantic models to dictionaries
        messages = [msg.dict() for msg in request.messages]
        
        # Call Eden AI using OpenAI SDK interface
        response = client.chat.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Return the full response
        return JSONResponse(content=response)
    
    except Exception as e:
        error_detail = str(e)
        print(f"Error in chat_completions: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

# Raw Eden AI endpoint for direct testing
@app.post("/eden/raw")
async def eden_raw(
    request: Dict[str, Any] = Body(...)
):
    """
    Send a raw request to Eden AI for testing
    """
    try:
        # Get API key
        headers = {"Authorization": f"Bearer {EDEN_AI_API_KEY}"}
        
        # Make direct request to Eden AI
        response = requests.post("https://api.edenai.run/v2/llm/chat", 
                                json=request, 
                                headers=headers)
        
        # Return the raw response
        return JSONResponse(content=response.json())
    
    except Exception as e:
        error_detail = str(e)
        print(f"Error in eden_raw: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)