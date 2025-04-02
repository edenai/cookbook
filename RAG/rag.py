from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import requests
import os
from pydantic import BaseModel, HttpUrl

# Load API key from environment
EDEN_AI_API_KEY = os.getenv("EDEN_AI_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNmUxYjk5OGUtM2E1Yy00ODVjLWE4YTItZWJjYTEyZWM4MzMwIiwidHlwZSI6ImFwaV90b2tlbiJ9.Sp3RrL1gAzT9Y5kMD6k2_ncwvIhcnpHd6yH6-Duuoio")
if not EDEN_AI_API_KEY:
    raise ValueError("Environment variable EDEN_AI_API_KEY is not set")

BASE_URL = "https://api.edenai.run/v2/aiproducts/askyoda/v2"

app = FastAPI(
    title="Eden AI RAG Interface",
    description="API for interacting with Eden AI's RAG capabilities",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to get headers with API key
def get_headers(content_type: str = "application/json"):
    return {
        "Authorization": f"Bearer {EDEN_AI_API_KEY}",
        "Content-Type": content_type,
        "Accept": "application/json",
    }

# Project creation models
class CreateProjectRequest(BaseModel):
    ocr_provider: str = "amazon"
    speech_to_text_provider: str = "openai"
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    project_name: str
    collection_name: str
    db_provider: str = "qdrant"
    embeddings_provider: str = "openai"
    chunk_size: Optional[int] = None
    chunk_separators: Optional[List[str]] = None

# Bot profile models
class CreateBotProfileRequest(BaseModel):
    model: str
    name: str
    text: str
    params: Optional[Dict[str, Any]] = None

# Add file models
class AddFileRequest(BaseModel):
    data_type: str
    file_url: Optional[str] = None
    metadata: Optional[str] = None
    provider: Optional[str] = None

# Add text models
class TextItem(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class AddTextsRequest(BaseModel):
    texts: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None

# Add URL models
class AddUrlsRequest(BaseModel):
    urls: List[str]  # Just use strings instead of HttpUrl
    js_render: Optional[List[bool]] = None
    metadata: Optional[List[Dict[str, Any]]] = None

# Ask LLM models
class MessageItem(BaseModel):
    user: str
    assistant: str

class AskLLMRequest(BaseModel):
    query: str
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    k: int = 3
    history: Optional[List[MessageItem]] = None
    chatbot_global_action: Optional[str] = None
    filter_documents: Optional[Dict[str, Any]] = None
    min_score: float = 0
    temperature: float = 0
    max_tokens: int = 100
    conversation_id: Optional[str] = None

# Create conversation models
class CreateConversationRequest(BaseModel):
    name: Optional[str] = None

# Query models (same as AskLLM)
class QueryRequest(AskLLMRequest):
    pass

# Project management
@app.post("/projects", summary="Create a new RAG project")
async def create_project(project_request: CreateProjectRequest):
    """Create a new RAG project with Eden AI."""
    try:
        payload = project_request.dict(exclude_none=True)
        print("Request payload:", payload)  # Debug print
        
        response = requests.post(
            f"{BASE_URL}/",
            headers=get_headers(),
            json=payload,
        )
        print("Response status:", response.status_code)  # Debug print
        print("Response text:", response.text)  # Debug print
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error creating project: {str(e)}")

@app.get("/projects", summary="List all RAG projects")
async def list_projects():
    """List all RAG projects."""
    try:
        response = requests.get(
            f"{BASE_URL}/",
            headers=get_headers(),
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error listing projects: {str(e)}")

@app.get("/projects/{project_id}", summary="Retrieve a RAG project")
async def get_project(project_id: str):
    """Retrieve details for a specific RAG project."""
    try:
        response = requests.get(
            f"{BASE_URL}/{project_id}/",
            headers=get_headers(),
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving project: {str(e)}")

@app.delete("/projects/{project_id}", summary="Delete a RAG project")
async def delete_project(project_id: str):
    """Delete a RAG project."""
    try:
        response = requests.delete(
            f"{BASE_URL}/{project_id}/",
            headers=get_headers(),
        )
        response.raise_for_status()
        return {"message": "Project deleted successfully"}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error deleting project: {str(e)}")

# Bot profile management
@app.post("/projects/{project_id}/bot-profile", summary="Create a bot profile")
async def create_bot_profile(project_id: str, profile_request: CreateBotProfileRequest):
    """Create a bot profile (system prompt) for a RAG project."""
    try:
        response = requests.post(
            f"{BASE_URL}/{project_id}/create_bot_prompt/",
            headers=get_headers(),
            json=profile_request.dict(exclude_none=True),
        )
        response.raise_for_status()
        return {"message": "Bot profile created successfully"}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error creating bot profile: {str(e)}")

@app.delete("/projects/{project_id}/bot-profile", summary="Remove a bot profile")
async def remove_bot_profile(project_id: str):
    """Remove the bot profile from a RAG project."""
    try:
        response = requests.delete(
            f"{BASE_URL}/{project_id}/remove_bot_prompt/",
            headers=get_headers(),
        )
        response.raise_for_status()
        return {"message": "Bot profile removed successfully"}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error removing bot profile: {str(e)}")

# Data upload - File
@app.post("/projects/{project_id}/files", summary="Upload a file")
async def upload_file(
    project_id: str,
    data_type: str = Form(...),
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    provider: Optional[str] = Form(None)
):
    """Upload a file to the RAG project."""
    if not file and not file_url:
        raise HTTPException(status_code=400, detail="Either file or file_url must be provided")
    
    try:
        if file:
            # Upload actual file
            files = {"file": (file.filename, await file.read(), file.content_type)}
            data = {
                "data_type": data_type,
            }
            if metadata:
                data["metadata"] = metadata
            if provider:
                data["provider"] = provider
                
            response = requests.post(
                f"{BASE_URL}/{project_id}/add_file/",
                headers={"Authorization": f"Bearer {EDEN_AI_API_KEY}"},
                data=data,
                files=files,
            )
        else:
            # Use file URL
            payload = {
                "data_type": data_type,
                "file_url": file_url,
            }
            if metadata:
                payload["metadata"] = metadata
            if provider:
                payload["provider"] = provider
                
            response = requests.post(
                f"{BASE_URL}/{project_id}/add_file/",
                headers=get_headers(),
                json=payload,
            )
            
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

# Data upload - Text
@app.post("/projects/{project_id}/texts", summary="Add text data")
async def add_texts(project_id: str, text_request: AddTextsRequest):
    """Add text data to the RAG project."""
    try:
        response = requests.post(
            f"{BASE_URL}/{project_id}/add_text/",
            headers=get_headers(),
            json=text_request.dict(exclude_none=True),
        )
        response.raise_for_status()
        return {"message": "Texts added successfully"}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error adding texts: {str(e)}")

# Data upload - URLs
@app.post("/projects/{project_id}/urls", summary="Add URLs")
async def add_urls(project_id: str, url_request: AddUrlsRequest):
    """Add URLs to the RAG project."""
    try:
        response = requests.post(
            f"{BASE_URL}/{project_id}/add_url/",
            headers=get_headers(),
            json=url_request.dict(exclude_none=True),
        )
        response.raise_for_status()
        return {"message": "URLs added successfully"}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error adding URLs: {str(e)}")

# Ask LLM
@app.post("/projects/{project_id}/ask-llm", summary="Ask the LLM")
async def ask_llm(project_id: str, llm_request: AskLLMRequest):
    """Ask the LLM based on the project data."""
    try:
        response = requests.post(
            f"{BASE_URL}/{project_id}/ask_llm/",
            headers=get_headers(),
            json=llm_request.dict(exclude_none=True),
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error asking LLM: {str(e)}")

# Conversation management
@app.post("/projects/{project_id}/conversations", summary="Create a conversation")
async def create_conversation(project_id: str, conversation_request: Optional[CreateConversationRequest] = None):
    """Create a new conversation for the RAG project."""
    try:
        payload = {} if conversation_request is None else conversation_request.dict(exclude_none=True)
        response = requests.post(
            f"{BASE_URL}/{project_id}/conversations/",
            headers=get_headers(),
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error creating conversation: {str(e)}")

@app.get("/projects/{project_id}/conversations", summary="List conversations")
async def list_conversations(project_id: str):
    """List all conversations for the RAG project."""
    try:
        response = requests.get(
            f"{BASE_URL}/{project_id}/conversations/",
            headers=get_headers(),
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error listing conversations: {str(e)}")

@app.get("/projects/{project_id}/conversations/{conversation_id}", summary="Get conversation details")
async def get_conversation(project_id: str, conversation_id: str):
    """Get details for a specific conversation."""
    try:
        response = requests.get(
            f"{BASE_URL}/{project_id}/conversations/{conversation_id}/",
            headers=get_headers(),
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")

@app.delete("/projects/{project_id}/conversations/{conversation_id}", summary="Delete a conversation")
async def delete_conversation(project_id: str, conversation_id: str):
    """Delete a conversation."""
    try:
        response = requests.delete(
            f"{BASE_URL}/{project_id}/conversations/{conversation_id}/",
            headers=get_headers(),
        )
        response.raise_for_status()
        return {"message": "Conversation deleted successfully"}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")

# Query (search)
@app.post("/projects/{project_id}/query", summary="Query the project data")
async def query_data(project_id: str, query_request: QueryRequest):
    """Query the project data using the LLM."""
    try:
        response = requests.post(
            f"{BASE_URL}/{project_id}/query/",
            headers=get_headers(),
            json=query_request.dict(exclude_none=True),
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error querying data: {str(e)}")

# Data management - Delete chunk
@app.delete("/projects/{project_id}/chunks", summary="Delete chunks")
async def delete_chunks(
    project_id: str,
    id: Optional[str] = None,
    chunk_ids: Optional[List[str]] = Query(None)
):
    """Delete one or more chunks from the project."""
    try:
        if id:
            # Delete a single chunk by ID (query parameter)
            response = requests.delete(
                f"{BASE_URL}/{project_id}/delete_chunk/?id={id}",
                headers=get_headers(),
            )
        elif chunk_ids:
            # Delete multiple chunks
            response = requests.delete(
                f"{BASE_URL}/{project_id}/delete_chunk/",
                headers=get_headers(),
                json={"chunk_ids": chunk_ids},
            )
        else:
            raise HTTPException(status_code=400, detail="Either id or chunk_ids must be provided")
            
        response.raise_for_status()
        return {"message": "Chunks deleted successfully"}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error deleting chunks: {str(e)}")

# Data management - Delete all chunks
@app.delete("/projects/{project_id}/all-chunks", summary="Delete all chunks")
async def delete_all_chunks(project_id: str):
    """Delete all chunks from the project."""
    try:
        response = requests.delete(
            f"{BASE_URL}/{project_id}/delete_all_chunks/",
            headers=get_headers(),
        )
        response.raise_for_status()
        return {"message": "All chunks deleted successfully"}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error deleting all chunks: {str(e)}")

# Data management - Delete file
@app.delete("/projects/{project_id}/files/{file_id}", summary="Delete a file")
async def delete_file(project_id: str, file_id: str):
    """Delete a file from the project."""
    try:
        response = requests.delete(
            f"{BASE_URL}/{project_id}/files/{file_id}/",
            headers=get_headers(),
        )
        response.raise_for_status()
        return {"message": "File deleted successfully"}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

# File listing
@app.get("/projects/{project_id}/files", summary="List files")
async def list_files(project_id: str):
    """List all files in the project."""
    try:
        response = requests.get(
            f"{BASE_URL}/{project_id}/files/",
            headers=get_headers(),
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)