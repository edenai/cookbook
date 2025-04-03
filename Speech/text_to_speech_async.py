import os
import requests
import json
from typing import Optional, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()
EDENAI_API_KEY = os.getenv("EDENAI_API_KEY")

app = FastAPI()

class TTSCreateRequest(BaseModel):
    providers: str = Field(..., description="Comma-separated providers, e.g. 'amazon,microsoft'")
    text: str = Field(..., min_length=1, description="Text to convert to speech")
    language: Optional[str] = None
    option: Optional[str] = None
    voice_id: Optional[Dict[str, str]] = None
    rate: int = 0
    pitch: int = 0
    volume: int = 0
    audio_format: Optional[str] = None
    sampling_rate: int = 0
    response_as_dict: bool = True
    attributes_as_list: bool = False
    show_base_64: bool = True
    show_original_response: bool = False
    fallback_providers: Optional[str] = None

@app.post("/tts")
def create_tts_job(payload: TTSCreateRequest):
    """
    Create an asynchronous TTS job on Eden AI. 
    Must provide either (language + option) or a voice_id for each provider.
    """
    # 1) Validate Eden AI TTS requirements
    if not payload.voice_id:
        if not payload.language or not payload.option:
            raise HTTPException(
                status_code=400,
                detail="You must provide either (language AND option) OR a voice_id for each provider."
            )
    else:
        providers_list = [p.strip() for p in payload.providers.split(",") if p.strip()]
        for prov in providers_list:
            if prov not in payload.voice_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"No voice_id specified for provider '{prov}'."
                )

    # 2) Convert fallback_providers to a list if present
    fallback_list = None
    if payload.fallback_providers:
        fallback_list = [fp.strip() for fp in payload.fallback_providers.split(",") if fp.strip()]

    # 3) Build the JSON body
    body = {
        "providers": payload.providers,
        "text": payload.text,
        "rate": payload.rate,
        "pitch": payload.pitch,
        "volume": payload.volume,
        "audio_format": payload.audio_format,
        "sampling_rate": payload.sampling_rate,
        "response_as_dict": payload.response_as_dict,
        "attributes_as_list": payload.attributes_as_list,
        "show_base_64": payload.show_base_64,
        "show_original_response": payload.show_original_response,
    }
    if fallback_list:
        body["fallback_providers"] = fallback_list
    if payload.voice_id:
        body["voice_id"] = payload.voice_id
    else:
        body["language"] = payload.language
        body["option"] = payload.option

    # 4) Send request to Eden AI
    url = "https://api.edenai.run/v2/audio/text_to_speech_async"
    headers = {
        "Authorization": f"Bearer {EDENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=body)
    return {
        "status_code": response.status_code,
        "response": response.json()
    }

@app.get("/tts/{job_id}")
def get_tts_job(job_id: str):
    """
    Retrieve (poll) the status/result of a previously created TTS job.
    If completed, you'll see a 'job_status': 'completed' and 'audio_resource_url' or base64 data.
    """
    url = f"https://api.edenai.run/v2/audio/text_to_speech_async/{job_id}"
    headers = {
        "Authorization": f"Bearer {EDENAI_API_KEY}"
    }
    response = requests.get(url, headers=headers)
    return {
        "status_code": response.status_code,
        "response": response.json()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
