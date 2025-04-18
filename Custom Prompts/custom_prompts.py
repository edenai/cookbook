from fastapi import FastAPI, HTTPException, Query, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any, Union
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
EDEN_AI_API_KEY = os.getenv("EDEN_AI_API_KEY")
if not EDEN_AI_API_KEY:
    raise ValueError("EDEN_AI_API_KEY environment variable not set. Please add it to your .env file.")

# Define Eden AI base URL
EDEN_AI_BASE_URL = "https://api.edenai.run/v2/prompts"

# Initialize FastAPI
app = FastAPI(title="Eden AI Prompts API Proxy", description="A FastAPI application for interfacing with Eden AI's prompts API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Headers for Eden AI API requests
def get_headers():
    return {
        "Authorization": f"Bearer {EDEN_AI_API_KEY}",
        "accept": "application/json",
        "content-type": "application/json"
    }

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "Eden AI Prompts API Proxy",
        "version": "1.0.0",
        "description": "A FastAPI application that integrates with Eden AI Prompts API"
    }

# List all prompts
@app.get("/v2/prompts/", tags=["Prompts"])
async def list_prompts(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100)
):
    try:
        response = requests.get(
            f"{EDEN_AI_BASE_URL}/",
            headers=get_headers(),
            params={"page": page, "page_size": page_size}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        else:
            status_code = 500
            error_detail = str(e)
        
        raise HTTPException(status_code=status_code, detail=error_detail)

# Create a prompt
@app.post("/v2/prompts/", tags=["Prompts"])
async def create_prompt(request: Request):
    try:
        # Get the request body as JSON
        body = await request.json()
        
        # Forward the request to Eden AI
        response = requests.post(
            f"{EDEN_AI_BASE_URL}/",
            headers=get_headers(),
            json=body
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        else:
            status_code = 500
            error_detail = str(e)
        
        raise HTTPException(status_code=status_code, detail=error_detail)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

# Get a specific prompt
@app.get("/v2/prompts/{name}/", tags=["Prompts"])
async def get_prompt(name: str):
    try:
        response = requests.get(
            f"{EDEN_AI_BASE_URL}/{name}/",
            headers=get_headers()
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        else:
            status_code = 500
            error_detail = str(e)
        
        raise HTTPException(status_code=status_code, detail=error_detail)

# Call a prompt
@app.post("/v2/prompts/{name}/", tags=["Prompts"])
async def call_prompt(name: str, request: Request):
    try:
        # Get the request body as JSON
        body = await request.json()
        
        # Forward the request to Eden AI
        response = requests.post(
            f"{EDEN_AI_BASE_URL}/{name}/",
            headers=get_headers(),
            json=body
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        else:
            status_code = 500
            error_detail = str(e)
        
        raise HTTPException(status_code=status_code, detail=error_detail)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

# Update a prompt (PUT)
@app.put("/v2/prompts/{name}/", tags=["Prompts"])
async def update_prompt(name: str, request: Request):
    try:
        # Get the request body as JSON
        body = await request.json()
        
        # Forward the request to Eden AI
        response = requests.put(
            f"{EDEN_AI_BASE_URL}/{name}/",
            headers=get_headers(),
            json=body
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        else:
            status_code = 500
            error_detail = str(e)
        
        raise HTTPException(status_code=status_code, detail=error_detail)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

# Update a prompt (PATCH)
@app.patch("/v2/prompts/{name}/", tags=["Prompts"])
async def patch_prompt(name: str, request: Request):
    try:
        # Get the request body as JSON
        body = await request.json()
        
        # Forward the request to Eden AI
        response = requests.patch(
            f"{EDEN_AI_BASE_URL}/{name}/",
            headers=get_headers(),
            json=body
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        else:
            status_code = 500
            error_detail = str(e)
        
        raise HTTPException(status_code=status_code, detail=error_detail)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

# Delete a prompt
@app.delete("/v2/prompts/{name}/", tags=["Prompts"])
async def delete_prompt(name: str):
    try:
        response = requests.delete(
            f"{EDEN_AI_BASE_URL}/{name}/",
            headers=get_headers()
        )
        response.raise_for_status()
        return Response(status_code=204)
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        else:
            status_code = 500
            error_detail = str(e)
        
        raise HTTPException(status_code=status_code, detail=error_detail)

# Prompt History endpoints
@app.get("/v2/prompts/{name}/history/", tags=["Prompt History"])
async def list_prompt_history(
    name: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100)
):
    try:
        response = requests.get(
            f"{EDEN_AI_BASE_URL}/{name}/history/",
            headers=get_headers(),
            params={"page": page, "page_size": page_size}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        else:
            status_code = 500
            error_detail = str(e)
        
        raise HTTPException(status_code=status_code, detail=error_detail)

@app.post("/v2/prompts/{name}/history/", tags=["Prompt History"])
async def create_prompt_history(name: str, request: Request):
    try:
        # Get the request body as JSON
        body = await request.json()
        
        # Forward the request to Eden AI
        response = requests.post(
            f"{EDEN_AI_BASE_URL}/{name}/history/",
            headers=get_headers(),
            json=body
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        else:
            status_code = 500
            error_detail = str(e)
        
        raise HTTPException(status_code=status_code, detail=error_detail)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

@app.get("/v2/prompts/{name}/history/{id}/", tags=["Prompt History"])
async def get_prompt_history(name: str, id: str):
    try:
        response = requests.get(
            f"{EDEN_AI_BASE_URL}/{name}/history/{id}/",
            headers=get_headers()
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        else:
            status_code = 500
            error_detail = str(e)
        
        raise HTTPException(status_code=status_code, detail=error_detail)

@app.put("/v2/prompts/{name}/history/{id}/", tags=["Prompt History"])
async def update_prompt_history(name: str, id: str, request: Request):
    try:
        # Get the request body as JSON
        body = await request.json()
        
        # Forward the request to Eden AI
        response = requests.put(
            f"{EDEN_AI_BASE_URL}/{name}/history/{id}/",
            headers=get_headers(),
            json=body
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        else:
            status_code = 500
            error_detail = str(e)
        
        raise HTTPException(status_code=status_code, detail=error_detail)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

@app.patch("/v2/prompts/{name}/history/{id}/", tags=["Prompt History"])
async def patch_prompt_history(name: str, id: str, request: Request):
    try:
        # Get the request body as JSON
        body = await request.json()
        
        # Forward the request to Eden AI
        response = requests.patch(
            f"{EDEN_AI_BASE_URL}/{name}/history/{id}/",
            headers=get_headers(),
            json=body
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        else:
            status_code = 500
            error_detail = str(e)
        
        raise HTTPException(status_code=status_code, detail=error_detail)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

@app.delete("/v2/prompts/{name}/history/{id}/", tags=["Prompt History"])
async def delete_prompt_history(name: str, id: str):
    try:
        response = requests.delete(
            f"{EDEN_AI_BASE_URL}/{name}/history/{id}/",
            headers=get_headers()
        )
        response.raise_for_status()
        from fastapi import Response
        return Response(status_code=204)
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        else:
            status_code = 500
            error_detail = str(e)
        
        raise HTTPException(status_code=status_code, detail=error_detail)

@app.get("/v2/prompts/{name}/history/{id}/template-variables/", tags=["Prompt History"])
async def get_prompt_template_variables(name: str, id: str):
    try:
        response = requests.get(
            f"{EDEN_AI_BASE_URL}/{name}/history/{id}/template-variables/",
            headers=get_headers()
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        else:
            status_code = 500
            error_detail = str(e)
        
        raise HTTPException(status_code=status_code, detail=error_detail)

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)