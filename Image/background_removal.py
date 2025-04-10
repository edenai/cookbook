from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import tempfile
import os
from dotenv import load_dotenv
from typing import Optional, List
from pydantic import BaseModel, HttpUrl

app = FastAPI(title="Background Removal API")

load_dotenv()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# API Key for EdenAI
EDENAI_API_KEY = os.getenv('EDEN_AI_API_KEY')
EDENAI_API_ENDPOINT = "https://api.edenai.run/v2/image/background_removal"

class ImageURL(BaseModel):
    file_url: str
    providers: List[str]
    fallback_providers: Optional[List[str]] = None
    response_as_dict: Optional[bool] = True
    attributes_as_list: Optional[bool] = False
    show_base_64: Optional[bool] = True
    show_original_response: Optional[bool] = False
    provider_params: Optional[dict] = None

@app.get("/")
def read_root():
    return {"message": "Background Removal API", "endpoints": ["/remove-bg/url", "/remove-bg/upload"]}

@app.post("/remove-bg/url")
async def remove_background_from_url(data: ImageURL):
    """
    Remove background from an image using its URL
    """
    headers = {"Authorization": f"Bearer {EDENAI_API_KEY}"}
    
    # Prepare payload from the model
    json_payload = {
        "providers": ",".join(data.providers),
        "file_url": data.file_url,
        "response_as_dict": data.response_as_dict,
        "attributes_as_list": data.attributes_as_list,
        "show_base_64": data.show_base_64,
        "show_original_response": data.show_original_response
    }
    
    # Add fallback_providers if provided
    if data.fallback_providers:
        json_payload["fallback_providers"] = ",".join(data.fallback_providers)
    
    # Add provider_params if provided
    if data.provider_params:
        json_payload["provider_params"] = data.provider_params
    
    try:
        response = requests.post(EDENAI_API_ENDPOINT, json=json_payload, headers=headers)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        result = response.json()
        
        return result
        
    except requests.exceptions.HTTPError as e:
        # Return the actual error response from EdenAI
        if e.response is not None:
            error_detail = e.response.json() if e.response.content else str(e)
            raise HTTPException(status_code=e.response.status_code, detail=error_detail)
        else:
            raise HTTPException(status_code=500, detail=f"Error communicating with EdenAI API: {str(e)}")
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with EdenAI API: {str(e)}")

@app.post("/remove-bg/upload")
async def remove_background_from_file(
    file: UploadFile = File(...),
    providers: str = Form(...),  # Comma-separated list of providers
    fallback_providers: Optional[str] = Form(None),  # Comma-separated list of fallback providers
    response_as_dict: Optional[bool] = Form(True),
    attributes_as_list: Optional[bool] = Form(False),
    show_base_64: Optional[bool] = Form(True),
    show_original_response: Optional[bool] = Form(False)
):
    """
    Remove background from an uploaded image file
    """
    headers = {"Authorization": f"Bearer {EDENAI_API_KEY}"}
    temp_file = None
    file_content = await file.read()
    
    try:
        # Create a temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".tmp")
        temp_file = os.fdopen(temp_fd, "wb")
        temp_file.write(file_content)
        temp_file.close()  # Important: Close the file before using it with requests
        
        # Prepare form data
        data = {
            "providers": providers,
            "response_as_dict": str(response_as_dict).lower(),
            "attributes_as_list": str(attributes_as_list).lower(),
            "show_base_64": str(show_base_64).lower(),
            "show_original_response": str(show_original_response).lower()
        }
        
        # Add fallback_providers if provided
        if fallback_providers:
            data["fallback_providers"] = fallback_providers
        
        # Properly use a context manager for the file
        with open(temp_path, "rb") as f:
            files = {"file": (file.filename, f, file.content_type or "application/octet-stream")}
            response = requests.post(EDENAI_API_ENDPOINT, data=data, files=files, headers=headers)
        
        # Handle response status
        if response.status_code != 200:
            error_detail = response.json() if response.content else "Unknown error"
            raise HTTPException(status_code=response.status_code, detail=error_detail)
            
        return response.json()
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with EdenAI API: {str(e)}")
    finally:
        # Close file handle if it's still open
        if temp_file and not temp_file.closed:
            temp_file.close()
            
        # Try to delete the temp file, but don't raise an exception if it fails
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except (PermissionError, OSError) as e:
                # On Windows, sometimes we can't delete immediately
                # Just log it (in a real app) and continue
                print(f"Warning: Could not delete temporary file {temp_path}: {e}")
                # In a production app, you might want to schedule it for deletion later

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)