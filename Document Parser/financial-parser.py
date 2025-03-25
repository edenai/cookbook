from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from typing import Optional, List
from pydantic import BaseModel, HttpUrl
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
FINANCIAL_PARSER_URL = "https://api.edenai.run/v2/ocr/financial_parser"

class FinancialParserResponse(BaseModel):
    provider_responses: dict
    status: str

@app.post("/parse-financial-file/", response_model=FinancialParserResponse)
async def parse_financial_file(
    file: UploadFile = File(None),
    file_url: Optional[str] = Form(None),
    providers: str = Form("amazon,base64,microsoft,mindee"),
    fallback_providers: Optional[str] = Form(None),
    response_as_dict: bool = Form(True),
    attributes_as_list: bool = Form(False),
    show_base64: bool = Form(True),
    show_original_response: bool = Form(False),
    file_password: Optional[str] = Form(None),
    language: Optional[str] = Form("en"),
    document_type: str = Form("invoice"),
    convert_to_pdf: bool = Form(False)
):
    try:
        # Validate input
        if not file and not file_url:
            raise HTTPException(
                status_code=400,
                detail="Either file or file_url must be provided"
            )
        if file and file_url:
            raise HTTPException(
                status_code=400,
                detail="Cannot provide both file and file_url"
            )

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {EDEN_AI_API_TOKEN}"
        }

        # Prepare the payload
        payload = {
            "providers": providers,
            "fallback_providers": fallback_providers,
            "response_as_dict": response_as_dict,
            "attributes_as_list": attributes_as_list,
            "show_base64": show_base64,
            "show_original_response": show_original_response,
            "language": language,
            "document_type": document_type,
            "convert_to_pdf": convert_to_pdf
        }

        if file_password:
            payload["file_password"] = file_password

        # Handle file upload vs URL
        if file:
            file_content = await file.read()
            files = {
                "file": (file.filename, file_content, file.content_type)
            }
            response = requests.post(
                FINANCIAL_PARSER_URL,
                data=payload,
                files=files,
                headers=headers
            )
        else:
            payload["file_url"] = file_url
            response = requests.post(
                FINANCIAL_PARSER_URL,
                json=payload,
                headers=headers
            )

        # Check if the request was successful
        response.raise_for_status()
        result = response.json()

        return {
            "provider_responses": result,
            "status": "success"
        }

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)