import os
import json
import requests
from fastapi import FastAPI, File, Form, UploadFile
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
EDENAI_API_KEY = os.getenv("EDENAI_API_KEY")

app = FastAPI()

@app.post("/parse-resume")
async def parse_resume(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    providers: str = Form("affinda"),              # e.g. "affinda", or "affinda,amazon"
    fallback_providers: Optional[str] = Form(None), # e.g. "google,amazon"
    settings: Optional[str] = Form(None),           # JSON string, e.g. '{"google":"model1"}'
    response_as_dict: bool = Form(True),
    attributes_as_list: bool = Form(False),
    show_base_64: bool = Form(True),
    show_original_response: bool = Form(False),
    file_password: Optional[str] = Form(None),
    convert_to_pdf: bool = Form(False)
):
    """
    Single endpoint that handles:
      - Local file uploads (multipart/form-data)
      - Remote file URLs (via form-data or JSON)
    Implements all documented parameters for Eden AI's Resume Parser.
    """

    # Validate input
    if not file and not file_url:
        return {"error": "Please provide either a 'file' or 'file_url'."}
    if file and file_url:
        return {"error": "Please provide only one: 'file' OR 'file_url', not both."}

    # Parse fallback_providers into a list if given (comma-separated)
    fallback_list = None
    if fallback_providers:
        fallback_list = [fp.strip() for fp in fallback_providers.split(",") if fp.strip()]

    # Parse settings if given (JSON string)
    settings_obj = None
    if settings:
        try:
            settings_obj = json.loads(settings)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in 'settings' parameter."}

    # Build common parameters
    common_params = {
        "providers": providers,
        "response_as_dict": response_as_dict,
        "attributes_as_list": attributes_as_list,
        "show_base_64": show_base_64,
        "show_original_response": show_original_response,
        "convert_to_pdf": convert_to_pdf
    }
    if fallback_list:
        # If fallback is used, the doc says only one provider can be in "providers"
        common_params["fallback_providers"] = fallback_list
    if settings_obj:
        common_params["settings"] = settings_obj
    if file_password:
        common_params["file_password"] = file_password

    url = "https://api.edenai.run/v2/ocr/resume_parser"
    headers = {"Authorization": f"Bearer {EDENAI_API_KEY}"}

    # CASE 1: Local File -> Use multipart/form-data
    if file:
        multipart_form_data = {}
        # Add each common param as (None, string_value) for multipart
        for key, value in common_params.items():
            if isinstance(value, bool):
                value = "true" if value else "false"
            elif isinstance(value, list) or isinstance(value, dict):
                value = json.dumps(value)
            multipart_form_data[key] = (None, str(value))
        # Attach the file
        multipart_form_data["file"] = (file.filename, file.file, file.content_type)
        response = requests.post(url, headers=headers, files=multipart_form_data)

    # CASE 2: Remote File URL -> Use JSON body
    else:
        json_payload = {**common_params, "file_url": file_url}
        response = requests.post(url, headers=headers, json=json_payload)

    return {
        "status_code": response.status_code,
        "response": response.json()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)