"""
Running conformer as a service
"""

from fastapi import FastAPI
from pydantic import BaseModel
from src.asr import ASRService

api = FastAPI()
service = ASRService()


class TranscribeRequest(BaseModel):
    audio_filepath: str | None = None


@api.get("/")
def root():
    """
    Test if API endpoint works
    """
    return {"response": "200", "message": "API service to do ASR with conformer"}


@api.post("/transcribe")
def transcribe(request: TranscribeRequest):
    print(request.json())
    return service.transcribe(request.audio_filepath)
